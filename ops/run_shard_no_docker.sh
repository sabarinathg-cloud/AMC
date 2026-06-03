#!/usr/bin/env bash
set -Eeuo pipefail

umask 0002

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
AMC_IN="${AMC_IN:-/mnt/amc-data}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required, for example /mnt/amc-runs/2026-smoke-100}"
HASH_MODE="${HASH_MODE:-path}"
NUM_SHARDS="${NUM_SHARDS:-}"
STAGES="${STAGES:-preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest}"
AUTO_PULL="${AUTO_PULL:-1}"

INSTANCE_ID="${INSTANCE_ID:-$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || hostname)}"
HOSTNAME_VALUE="$(hostname)"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
  write_status "repo_pull" "running" "updating shared checkout"
  (
    flock -w 900 9
    git config --global --add safe.directory "$REPO_DIR" || true
    git pull --ff-only
  ) 9>"$LOCK_DIR/repo_pull.lock"
fi

read -r RUN_FILE_COUNT RUN_SIGNATURE < <(
  "$PYTHON_BIN" ops/input_signature.py \
    --input "$AMC_IN" \
    --output "$AMC_OUT" \
    --hash-mode "$HASH_MODE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
)

write_status "input_signature" "ready" "files=$RUN_FILE_COUNT signature=$RUN_SIGNATURE"

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

  write_status "$key" "running" ""
  {
    echo
    echo "===== $(date -Is) START $key shard=$SHARD_INDEX/$NUM_SHARDS instance=$INSTANCE_ID host=$HOSTNAME_VALUE ====="
    nvidia-smi || true
    free -h || true
  } 2>&1 | tee -a "$log"

  set +e
  "$PYTHON_BIN" -m amc_pipeline.cli run-stage "$@" \
    --input "$AMC_IN" \
    --output "$AMC_OUT" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --discovery-hash-mode "$HASH_MODE" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e

  rm -rf "$AMC_OUT/.pii_pipeline/temp"/* 2>/dev/null || true
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
      run_stage asr_whisper asr --models whisper
      ;;
    asr_qwen)
      run_stage asr_qwen asr --models qwen --asr-batch-sizes qwen=1
      ;;
    asr_cohere)
      run_stage asr_cohere asr --models cohere --asr-batch-sizes cohere=1
      ;;
    asr_granite)
      run_stage asr_granite asr --models granite --asr-batch-sizes granite=1
      ;;
    normalize)
      run_stage normalize normalize
      ;;
    consensus)
      run_stage consensus consensus
      ;;
    pii)
      run_stage pii pii --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
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
      run_stage manifest manifest
      ;;
    *)
      echo "Unknown stage key: $stage" >&2
      exit 2
      ;;
  esac
done

write_status "complete" "completed" "all requested stages completed"
echo "Shard $SHARD_INDEX complete: $AMC_OUT"
