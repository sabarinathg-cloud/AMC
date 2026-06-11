#!/usr/bin/env bash
set -Eeuo pipefail

umask 0002

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
AMC_IN="${AMC_IN:-/mnt/amc-data}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required, for example /mnt/amc-data/amc-runs/2026-smoke-100}"
HASH_MODE="${HASH_MODE:-path}"
NUM_SHARDS="${NUM_SHARDS:-}"
STAGES="${STAGES:-preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest}"
AUTO_PULL="${AUTO_PULL:-1}"

# Optional fleet-wide ASR batch override, e.g. AMC_ASR_BATCH_SIZES="whisper=64,qwen=64,cohere=8,granite=8".
# Passed as --asr-batch-sizes to every asr stage; only the model selected by --models is run, so
# the same string is safe for all four. Use the values recommended by ops/asr_batch_sweep.py.
# (${arr[@]+...} keeps this empty-safe under `set -u`.)
ASR_BATCH_ARGS=()
if [[ -n "${AMC_ASR_BATCH_SIZES:-}" ]]; then
  ASR_BATCH_ARGS=(--asr-batch-sizes "$AMC_ASR_BATCH_SIZES")
fi

INSTANCE_ID="${INSTANCE_ID:-$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || hostname)}"
HOSTNAME_VALUE="$(hostname)"

# --- Per-stage Python environments (isolated venvs on shared storage) ---
# Different stages need conflicting dependency versions, so they run in separate venvs
# built by ops/setup_env.sh: the `align` stage uses the whisperx venv; every other stage
# uses the main venv. If the venvs are not present yet (un-provisioned host), fall back to
# PYTHON_BIN so a single-env setup still works.
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
MAIN_PY="$VENV_ROOT/main/bin/python"
ALIGN_PY="$VENV_ROOT/align/bin/python"
COHERE_PY="$VENV_ROOT/cohere/bin/python"
[[ -x "$MAIN_PY" ]] || MAIN_PY="$PYTHON_BIN"
[[ -x "$ALIGN_PY" ]] || ALIGN_PY="$MAIN_PY"
# Cohere ASR needs transformers>=5.4.0 (CohereAsrForConditionalGeneration), which is
# incompatible with the main venv's transformers==4.57.6 (hard-pinned by qwen-asr). It runs
# in its own isolated venv. If that venv is absent, fall back to MAIN_PY -- the cohere model
# will simply error per-segment (ImportError) and the run degrades to a 3-model consensus,
# exactly the prior behavior, instead of failing the stage.
[[ -x "$COHERE_PY" ]] || COHERE_PY="$MAIN_PY"

stage_python() {  # $1 = run_stage key
  case "$1" in
    align) printf '%s' "$ALIGN_PY" ;;
    asr_cohere) printf '%s' "$COHERE_PY" ;;
    *) printf '%s' "$MAIN_PY" ;;
  esac
}

if [[ -z "${SHARD_INDEX:-}" ]]; then
  if [[ -z "${INSTANCE_IDS:-}" ]]; then
    echo "SHARD_INDEX or INSTANCE_IDS is required" >&2
    exit 2
  fi
  idx=0
  found=0
  for id in $INSTANCE_IDS; do
    if [[ "$id" == "$INSTANCE_ID" ]]; then
      SHARD_INDEX="$idx"
      found=1
      break
    fi
    idx=$((idx + 1))
  done
  if [[ "$found" != "1" ]]; then
    echo "Instance $INSTANCE_ID is not present in INSTANCE_IDS: $INSTANCE_IDS" >&2
    exit 2
  fi
fi

if [[ -z "$NUM_SHARDS" ]]; then
  NUM_SHARDS="$(wc -w <<< "${INSTANCE_IDS:-$SHARD_INDEX}")"
fi

AMC_OUT="${AMC_OUT:-$RUN_ROOT/outputs/shard-$SHARD_INDEX}"
LOG_DIR="$RUN_ROOT/logs/shard-$SHARD_INDEX"
STATUS_DIR="$RUN_ROOT/status"
LOCK_DIR="$RUN_ROOT/locks"
MARKER_DIR="$AMC_OUT/.pii_pipeline/stage_markers"
mkdir -p "$AMC_OUT" "$LOG_DIR" "$STATUS_DIR" "$LOCK_DIR" "$MARKER_DIR"

# Shared discovery cache (one tree walk per run instead of one per stage per box).
# Every stage's `amc_pipeline.cli run-stage` child inherits this; discover_audio_files
# builds the per-shard manifest once under a cross-host lock and everyone reads their
# slice. Records are reconstructed identically (same file_id/shard/signature), and any
# failure falls back to a live walk. Set AMC_DISCOVERY_CACHE=0 to disable.
export AMC_DISCOVERY_CACHE_DIR="${AMC_DISCOVERY_CACHE_DIR:-$RUN_ROOT/discovery-cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# CTranslate2 (faster-whisper / whisperx backend) dlopen's cuDNN/cuBLAS at runtime, but
# the pip-installed CUDA libs live under <venv>/site-packages/nvidia/*/lib which is not on
# the default loader path. Each venv ships its own matching cuDNN 9 build, so we resolve
# these dirs PER interpreter (in run_stage) rather than once globally.
# (CTranslate2 >= 4.5.0 links cuDNN 9, matching the torch cu121 wheel.)
_BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
nvidia_lib_dirs() {  # $1 = python interpreter
  "$1" - <<'PY' 2>/dev/null || true
import os
dirs = []
for mod_name in ("nvidia.cudnn", "nvidia.cublas"):
    try:
        mod = __import__(mod_name, fromlist=["__file__"])
        lib = os.path.join(os.path.dirname(mod.__file__), "lib")
        if os.path.isdir(lib):
            dirs.append(lib)
    except Exception:
        pass
print(":".join(dirs))
PY
}
# Local scratch for decode/mask WAVs: keep heavy temp I/O off shared NFS.
export AMC_TEMP_DIR="${AMC_TEMP_DIR:-/tmp/amc-scratch/$INSTANCE_ID/shard-$SHARD_INDEX}"
mkdir -p "$AMC_TEMP_DIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/amc-cache/$INSTANCE_ID/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/amc-cache/$INSTANCE_ID/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

cd "$REPO_DIR"

write_status() {
  local stage="$1"
  local state="$2"
  local message="${3:-}"
  "$PYTHON_BIN" - "$STATUS_DIR/shard-$SHARD_INDEX.json" <<PY
import json, pathlib, time
path = pathlib.Path("$STATUS_DIR/shard-$SHARD_INDEX.json")
payload = {
    "shard_index": int("$SHARD_INDEX"),
    "num_shards": int("$NUM_SHARDS"),
    "stage": "$stage",
    "state": "$state",
    "message": "$message",
    "instance_id": "$INSTANCE_ID",
    "hostname": "$HOSTNAME_VALUE",
    "input": "$AMC_IN",
    "output": "$AMC_OUT",
    "input_files": "${RUN_FILE_COUNT:-}",
    "input_signature": "${RUN_SIGNATURE:-}",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if [[ "$AUTO_PULL" == "1" ]]; then
  # On a SHARED (NFS) checkout, concurrent pulls from N hosts race and corrupt
  # refs ("cannot lock ref refs/remotes/origin/main"); flock does not reliably
  # serialize across hosts on NFS. So only the lowest-index shard pulls, and the
  # rest wait for the marker. SSM runs as root without HOME, which breaks git
  # global config / ref locking, so pin HOME explicitly.
  export HOME="${HOME:-/root}"
  PULL_DONE_MARKER="$STATUS_DIR/.repo_pull_done"
  if [[ "$SHARD_INDEX" == "0" ]]; then
    write_status "repo_pull" "running" "updating shared checkout (shard 0)"
    rm -f "$PULL_DONE_MARKER" 2>/dev/null || true
    git config --global --add safe.directory "$REPO_DIR" || true
    git fetch origin
    git reset --hard origin/main
    git rev-parse HEAD > "$PULL_DONE_MARKER"
  else
    write_status "repo_pull" "waiting" "waiting for shard 0 to update shared checkout"
    for _ in $(seq 1 180); do
      [[ -f "$PULL_DONE_MARKER" ]] && break
      sleep 2
    done
  fi
fi

# --- HSM pre-warm gate (distributed, per-shard + fleet barrier) --------------
# On FSx for Lustre, cold input is released to S3 (lfs hsm_state = "released");
# the first read of a released file blocks in the kernel (ldlm_completion_ast)
# until S3 restores it. The discovery builder reads EVERY metadata.json's content,
# so a released tree wedges the single builder behind ~one-restore-per-file. So we
# restore the input to Lustre BEFORE any stage (or discovery) reads it.
#
# A single box restoring the whole tree is itself a ~1h serial metadata walk, so we
# distribute it: each shard restores only its 1/NUM_SHARDS slice of the top-level
# dirs (ls + NR%N==i), all boxes in parallel. The union of slices is the whole tree.
# Because the discovery builder walks the WHOLE tree (any box may win the build lock),
# every box then BARRIERS until all NUM_SHARDS slices report done -- only then is the
# whole tree guaranteed resident and safe to walk. Markers live in STATUS_DIR (shared
# Lustre), so they are visible fleet-wide. Restores are filesystem-wide, so a file
# restored by one box is resident for whichever box later processes it.
# AMC_PREWARM: auto (default, warm iff lfs present) | 1 (force) | 0 (off).
if [[ "${AMC_PREWARM:-auto}" != "0" ]]; then
  PREWARM_MARKER="$STATUS_DIR/.prewarm_done_$SHARD_INDEX"
  PREWARM_STATUS_FILE="$STATUS_DIR/prewarm-$SHARD_INDEX.json"
  write_status "prewarm" "running" "restoring slice $SHARD_INDEX/$NUM_SHARDS of input from S3 to Lustre"
  rm -f "$PREWARM_MARKER" 2>/dev/null || true
  if AMC_PREWARM_STATUS="$PREWARM_STATUS_FILE" \
     AMC_PREWARM_NUM_SHARDS="$NUM_SHARDS" \
     AMC_PREWARM_SHARD_INDEX="$SHARD_INDEX" \
     bash "$REPO_DIR/ops/prewarm_hsm.sh" "$AMC_IN"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$PREWARM_MARKER"
  else
    # Do not block the run forever on a partial restore; record it and proceed.
    # Stage reads (and the progress-gated discovery heartbeat) handle stragglers.
    echo "WARNING: prewarm slice $SHARD_INDEX incomplete; proceeding (stragglers restore on read)" >&2
    date -u +%Y-%m-%dT%H:%M:%SZ > "$PREWARM_MARKER"
  fi
  # Fleet barrier: the discovery builder reads every metadata.json across the WHOLE
  # tree, so ALL slices must be resident before ANY box discovers. Wait for all
  # NUM_SHARDS slice markers (or time out and proceed -- on-read restore + the
  # progress-gated builder heartbeat keep a missing slice from hanging the fleet).
  write_status "prewarm" "waiting" "waiting for all $NUM_SHARDS prewarm slices to finish"
  for _ in $(seq 1 "${AMC_PREWARM_BARRIER_TRIES:-2400}"); do
    n="$(ls -1 "$STATUS_DIR"/.prewarm_done_* 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n:-0}" -ge "$NUM_SHARDS" ]] && break
    sleep 10
  done
fi

read -r RUN_FILE_COUNT RUN_SIGNATURE RUN_TOTAL_COUNT < <(
  "$MAIN_PY" ops/input_signature.py \
    --input "$AMC_IN" \
    --output "$AMC_OUT" \
    --hash-mode "$HASH_MODE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
)
# Sanitize: if input_signature errored, the fields are empty/garbage. Treat as zero so the
# guards below catch it instead of crashing on a non-numeric arithmetic comparison.
[[ "${RUN_FILE_COUNT:-}"  =~ ^[0-9]+$ ]] || RUN_FILE_COUNT=0
[[ "${RUN_TOTAL_COUNT:-}" =~ ^[0-9]+$ ]] || RUN_TOTAL_COUNT=0

# Fail fast on a globally-empty input. If THIS host discovers zero audio files under AMC_IN,
# the input is almost certainly missing or on a NON-shared path -- e.g. /mnt/amc-runs is LOCAL
# per instance on this fleet, so only the host that created the subset would see it while every
# other worker silently no-ops all 13 stages. Make that loud instead of invisible.
if (( RUN_TOTAL_COUNT == 0 )); then
  write_status "input_signature" "failed" "discovered 0 audio files under AMC_IN=$AMC_IN on host=$HOSTNAME_VALUE -- is AMC_IN on the shared Lustre mount (/mnt/amc-data)? (/mnt/amc-runs is local per instance)"
  echo "FATAL: discovered 0 audio files under AMC_IN=$AMC_IN on host=$HOSTNAME_VALUE shard=$SHARD_INDEX/$NUM_SHARDS." >&2
  echo "       AMC_IN must live on the shared Lustre mount (/mnt/amc-data). /mnt/amc-runs is LOCAL per instance." >&2
  exit 3
fi

# A shard can legitimately draw zero files (e.g. 2 files across 4 shards). That is not an
# error -- record it clearly and exit 0 instead of churning all stages as no-ops.
if (( RUN_FILE_COUNT == 0 )); then
  write_status "input_signature" "empty" "shard $SHARD_INDEX/$NUM_SHARDS drew 0 of $RUN_TOTAL_COUNT discovered files; nothing to do"
  echo "[$(date -Is)] shard $SHARD_INDEX/$NUM_SHARDS has 0 of $RUN_TOTAL_COUNT files; exiting 0 (no work for this shard)"
  exit 0
fi

write_status "input_signature" "ready" "files=$RUN_FILE_COUNT of total=$RUN_TOTAL_COUNT signature=$RUN_SIGNATURE"

run_stage() {
  local key="$1"
  shift
  local marker="$MARKER_DIR/$key.done"
  local log="$LOG_DIR/$key.log"
  if [[ -f "$marker" && "${FORCE_STAGE:-0}" != "1" ]]; then
    local marker_signature
    marker_signature="$(cat "$marker" 2>/dev/null || true)"
    if [[ "$marker_signature" == "$RUN_SIGNATURE" ]]; then
      write_status "$key" "skipped" "stage marker already exists for files=$RUN_FILE_COUNT signature=$RUN_SIGNATURE"
      echo "[$(date -Is)] SKIP $key signature=$RUN_SIGNATURE files=$RUN_FILE_COUNT" | tee -a "$log"
      return 0
    fi
    echo "[$(date -Is)] RERUN $key input changed old_signature=$marker_signature new_signature=$RUN_SIGNATURE files=$RUN_FILE_COUNT" | tee -a "$log"
  fi

  local py ld
  py="$(stage_python "$key")"
  ld="$(nvidia_lib_dirs "$py")"
  if [[ -n "$ld" ]]; then
    ld="$ld:$_BASE_LD_LIBRARY_PATH"
  else
    ld="$_BASE_LD_LIBRARY_PATH"
  fi

  write_status "$key" "running" ""
  {
    echo
    echo "===== $(date -Is) START $key shard=$SHARD_INDEX/$NUM_SHARDS instance=$INSTANCE_ID host=$HOSTNAME_VALUE ====="
    echo "interpreter: $py"
    nvidia-smi || true
    free -h || true
  } 2>&1 | tee -a "$log"

  set +e
  LD_LIBRARY_PATH="$ld" "$py" -m amc_pipeline.cli run-stage "$@" \
    --input "$AMC_IN" \
    --output "$AMC_OUT" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --discovery-hash-mode "$HASH_MODE" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e

  rm -rf "$AMC_TEMP_DIR"/* "$AMC_OUT/.pii_pipeline/temp"/* 2>/dev/null || true
  sync || true
  echo "===== $(date -Is) END $key rc=$rc =====" | tee -a "$log"
  if [[ "$rc" == "0" ]]; then
    echo "$RUN_SIGNATURE" > "$marker"
    write_status "$key" "completed" ""
  else
    write_status "$key" "failed" "exit code $rc; see $log"
    exit "$rc"
  fi
}

for stage in $STAGES; do
  case "$stage" in
    preprocess)
      run_stage preprocess preprocess --vad-backend silero
      ;;
    asr_whisper)
      run_stage asr_whisper asr --models whisper ${ASR_BATCH_ARGS[@]+"${ASR_BATCH_ARGS[@]}"}
      ;;
    asr_qwen)
      # Dynamic duration-budgeted batching (config defaults: count cap 8, ~240s budget).
      # Override per run with AMC_ASR_BATCH_SIZES (see top) / --asr-batch-sizes qwen=N, or via
      # config asr_models.qwen.{batch_audio_sec_budget,max_batch_size}. Size it with ops/asr_batch_sweep.py.
      run_stage asr_qwen asr --models qwen ${ASR_BATCH_ARGS[@]+"${ASR_BATCH_ARGS[@]}"}
      ;;
    asr_cohere)
      # Dynamic batching (config defaults: count cap 4, ~160s budget). float32 by default;
      # set asr_models.cohere.dtype: bfloat16 only after ops/asr_parity_check.py passes.
      run_stage asr_cohere asr --models cohere ${ASR_BATCH_ARGS[@]+"${ASR_BATCH_ARGS[@]}"}
      ;;
    asr_granite)
      # Dynamic batching (config defaults: count cap 4, ~160s budget); bf16 + SDPA on CUDA.
      run_stage asr_granite asr --models granite ${ASR_BATCH_ARGS[@]+"${ASR_BATCH_ARGS[@]}"}
      ;;
    normalize)
      run_stage normalize normalize
      ;;
    consensus)
      run_stage consensus consensus
      ;;
    pii)
      run_stage pii pii --detectors regex,piiranha,spacy,rule_name,saved_json
      ;;
    align)
      run_stage align align
      ;;
    mask_plan)
      run_stage mask_plan mask-plan
      ;;
    redact)
      run_stage redact redact --mask-strategy beep --allow-fallback-format wav
      ;;
    validate)
      run_stage validate validate
      ;;
    manifest)
      # At scale (default) skip the multi-GB single CSV and the per-year duplicates; JSONL stays
      # as the durable record and Parquet as the dataset (ops/merge_shards.sh reads those).
      # Set AMC_MANIFEST_LEAN=0 to restore CSV + per-year for small/manual runs.
      if [[ "${AMC_MANIFEST_LEAN:-1}" == "1" ]]; then
        run_stage manifest manifest --manifest-no-csv --manifest-no-per-year
      else
        run_stage manifest manifest
      fi
      ;;
    *)
      echo "Unknown stage key: $stage" >&2
      exit 2
      ;;
  esac
done

write_status "complete" "completed" "all requested stages completed"
echo "Shard $SHARD_INDEX complete: $AMC_OUT"
