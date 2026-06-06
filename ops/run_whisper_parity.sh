#!/usr/bin/env bash
# Whisper packing A/B on a GPU host: per-segment (AMC_WHISPER_BATCHED=0) vs the
# default packed path, on segments already in a run's state DB. Reports transcript
# drift + RTFx (speed) for each path. Run after preprocess has produced segments.
#
# Usage (host or via SSM):
#   bash ops/run_whisper_parity.sh
#   LIMIT=120 bash ops/run_whisper_parity.sh
#   AMC_OUT=/mnt/amc-data/amc-runs/<run>/outputs/shard-0 bash ops/run_whisper_parity.sh
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> SHARD_INDEX=0 bash ops/run_whisper_parity.sh
#
# Optionally A/B the turbo model too: fetch it first (ops/fetch_whisper_model.sh),
# then set WHISPER_PATH=/mnt/amc-data/pipeline/models/whisper-large-v3-turbo here.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
AMC_IN="${AMC_IN:-/mnt/amc-data}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
MAIN_PY="$VENV_ROOT/main/bin/python"
LIMIT="${LIMIT:-60}"

[[ -x "$MAIN_PY" ]] || { echo "FATAL: main venv python not found at $MAIN_PY (run ops/setup_env.sh)"; exit 2; }
cd "$REPO_DIR"

resolve_out() {
  if [[ -n "${AMC_OUT:-}" ]]; then printf '%s' "$AMC_OUT"; return; fi
  if [[ -n "${RUN_ROOT:-}" && -n "${SHARD_INDEX:-}" ]]; then
    printf '%s' "$RUN_ROOT/outputs/shard-$SHARD_INDEX"; return
  fi
  "$MAIN_PY" - "$AMC_IN" <<'PY'
import glob, os, sqlite3, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/amc-data"
suffix = os.path.join("outputs", "shard-*", ".pii_pipeline", "state", "pipeline.sqlite3")
patterns = [
    os.path.join(root, "amc-runs", "*", suffix),
    os.path.join(root, "*", suffix),
    os.path.join(root, suffix),
    os.path.join(root, "*", "*", suffix),
]
seen, candidates = set(), []
for pat in patterns:
    for db in glob.glob(pat):
        if db not in seen:
            seen.add(db)
            candidates.append(db)
best, best_n = None, 0
for db in candidates:
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        n = con.execute("select count(*) from segments").fetchone()[0]
        con.close()
    except Exception:
        n = 0
    if n > best_n:
        best, best_n = db, n
if best:
    out = os.path.dirname(os.path.dirname(os.path.dirname(best)))
    sys.stderr.write(f"auto-selected run with {best_n} segments: {out}\n")
    print(out)
PY
}

OUT="$(resolve_out)"
if [[ -z "$OUT" ]]; then
  echo "FATAL: no run with segments under $AMC_IN. Set AMC_OUT or RUN_ROOT+SHARD_INDEX," \
       "or run the preprocess stage first." >&2
  exit 3
fi
echo "parity output dir: $OUT"

# faster-whisper / ctranslate2 dlopen cuDNN+cuBLAS from the venv at runtime.
nvidia_lib_dirs() {
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
ld="$(nvidia_lib_dirs "$MAIN_PY")"
[[ -n "$ld" ]] && ld="$ld:${LD_LIBRARY_PATH:-}" || ld="${LD_LIBRARY_PATH:-}"

LOG_DIR="$OUT/.pii_pipeline/reports"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/whisper_parity_${STAMP}.log"
echo "writing log to: $LOG"

set +e
LD_LIBRARY_PATH="$ld" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$MAIN_PY" ops/asr_parity_check.py \
    --input "$AMC_IN" --output "$OUT" \
    --model whisper --limit "$LIMIT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "full log: $LOG"
exit "$rc"
