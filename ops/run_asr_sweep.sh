#!/usr/bin/env bash
# Run the ASR batch-size sweep (ops/asr_batch_sweep.py) for one or more models on a GPU host.
#
# It picks the right venv per model (cohere -> cohere venv, the rest -> main), resolves the
# per-venv cuDNN/cuBLAS lib dirs onto LD_LIBRARY_PATH (faster-whisper/ctranslate2 dlopen's
# them at runtime), points the sweep at a run whose state DB already has segments, and runs
# each model longest-segments-first with --verify so a bigger batch is only recommended when
# the transcripts are byte-identical.
#
# Usage (on the host, or via SSM):
#   bash ops/run_asr_sweep.sh                       # sweep all four models, auto-find a run
#   MODELS="qwen cohere" bash ops/run_asr_sweep.sh  # only these models
#   AMC_OUT=/mnt/amc-data/amc-runs/<run>/outputs/shard-0 bash ops/run_asr_sweep.sh
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> SHARD_INDEX=0 bash ops/run_asr_sweep.sh
#
# Per-model batch caps (override via env): SWEEP_CAPS_<MODEL>, e.g. SWEEP_CAPS_QWEN="8,16,32,64".
# LIMIT controls how many (longest) segments to benchmark (default 96).

# SSM's AWS-RunShellScript defaults to /bin/sh (dash) on some AMIs, which lacks `set -o pipefail`
# and bash arrays. Re-exec under bash so the rest of the script is portable.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
AMC_IN="${AMC_IN:-/mnt/amc-data}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
MAIN_PY="$VENV_ROOT/main/bin/python"
COHERE_PY="$VENV_ROOT/cohere/bin/python"
MODELS="${MODELS:-whisper qwen granite cohere}"
LIMIT="${LIMIT:-96}"

[[ -x "$MAIN_PY" ]] || { echo "FATAL: main venv python not found at $MAIN_PY (run ops/setup_env.sh)"; exit 2; }
cd "$REPO_DIR"

# --- Locate a run output dir whose state DB actually has segments -----------------------------
resolve_out() {
  if [[ -n "${AMC_OUT:-}" ]]; then printf '%s' "$AMC_OUT"; return; fi
  if [[ -n "${RUN_ROOT:-}" && -n "${SHARD_INDEX:-}" ]]; then
    printf '%s' "$RUN_ROOT/outputs/shard-$SHARD_INDEX"; return
  fi
  # Auto-discover: among all state DBs, pick the one with the most segments.
  "$MAIN_PY" - "$AMC_IN" <<'PY'
import os, sqlite3, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/amc-data"
best, best_n = None, 0
for dirpath, dirnames, filenames in os.walk(root):
    if "pipeline.sqlite3" in filenames and os.sep + "outputs" + os.sep in dirpath + os.sep:
        db = os.path.join(dirpath, "pipeline.sqlite3")
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            n = con.execute("select count(*) from segments").fetchone()[0]
            con.close()
        except Exception:
            n = 0
        if n > best_n:
            best, best_n = db, n
    # Do not descend into model weights / cache trees (speed + avoid huge dirs).
    dirnames[:] = [d for d in dirnames if d not in (".git", "models", "cache", "huggingface", "torch")]
if best:
    # db path is <out>/.pii_pipeline/state/pipeline.sqlite3 -> strip the trailing 3 components.
    out = os.path.dirname(os.path.dirname(os.path.dirname(best)))
    sys.stderr.write(f"auto-selected run with {best_n} segments: {out}\n")
    print(out)
PY
}

OUT="$(resolve_out)"
if [[ -z "$OUT" ]]; then
  echo "FATAL: no run found with segments under $AMC_IN. Set AMC_OUT or RUN_ROOT+SHARD_INDEX," \
       "or run the preprocess stage first." >&2
  exit 3
fi
echo "sweep output dir: $OUT"
echo "state db: $OUT/.pii_pipeline/state/pipeline.sqlite3"

# --- Per-venv cuDNN/cuBLAS lib dirs (faster-whisper / ctranslate2) ----------------------------
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

# Default caps per model (count-cap sweeps; the sweep also reports OOM-split + losslessness).
default_caps() {
  case "$1" in
    whisper) echo "16,32,48,64,96" ;;
    qwen)    echo "8,16,32,48,64" ;;
    cohere)  echo "4,8,16,24,32" ;;
    granite) echo "4,8,16,24,32" ;;
    *)       echo "4,8,16,32" ;;
  esac
}

LOG_DIR="$OUT/.pii_pipeline/reports"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$LOG_DIR/asr_batch_sweep_${STAMP}.log"
echo "writing full sweep log to: $SUMMARY"

for model in $MODELS; do
  py="$MAIN_PY"
  if [[ "$model" == "cohere" ]]; then
    if [[ -x "$COHERE_PY" ]]; then py="$COHERE_PY"; else
      echo "WARN: cohere venv missing at $COHERE_PY; skipping cohere" | tee -a "$SUMMARY"; continue
    fi
  fi
  # Per-model caps override: SWEEP_CAPS_QWEN, SWEEP_CAPS_WHISPER, ...
  upper="$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')"
  caps_var="SWEEP_CAPS_${upper}"
  caps="${!caps_var:-$(default_caps "$model")}"

  ld="$(nvidia_lib_dirs "$py")"
  [[ -n "$ld" ]] && ld="$ld:${LD_LIBRARY_PATH:-}" || ld="${LD_LIBRARY_PATH:-}"

  {
    echo
    echo "########################################################################"
    echo "# $(date -Is)  model=$model  interpreter=$py  caps=$caps  limit=$LIMIT"
    echo "########################################################################"
  } | tee -a "$SUMMARY"

  set +e
  LD_LIBRARY_PATH="$ld" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$py" ops/asr_batch_sweep.py \
      --input "$AMC_IN" --output "$OUT" \
      --model "$model" --limit "$LIMIT" --caps "$caps" --verify 2>&1 | tee -a "$SUMMARY"
  rc=${PIPESTATUS[0]}
  set -e
  [[ "$rc" == "0" ]] || echo "WARN: sweep for $model exited rc=$rc (continuing)" | tee -a "$SUMMARY"
done

echo
echo "=================== RECOMMENDATIONS ==================="
grep -E "^RECOMMEND|model=|apply via" "$SUMMARY" || true
echo "======================================================="
echo "full log: $SUMMARY"
