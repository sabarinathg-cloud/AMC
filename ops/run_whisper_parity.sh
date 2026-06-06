#!/usr/bin/env bash
# Whisper A/B on a GPU host, on segments already in a run's state DB. Reports
# transcript drift + RTFx (speed). Run after preprocess has produced segments.
#
# Two modes:
#   default            -- per-segment (WER-safe) vs packed (fast, opt-in)
#   WHISPER_PATH=<dir>  -- per-segment large-v3 vs per-segment <dir> (e.g. turbo),
#                          the WER-safe speed lever (#6).
#
# Usage (host or via SSM):
#   bash ops/run_whisper_parity.sh
#   LIMIT=120 bash ops/run_whisper_parity.sh
#   AMC_OUT=/mnt/amc-data/amc-runs/<run>/outputs/shard-0 bash ops/run_whisper_parity.sh
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> SHARD_INDEX=0 bash ops/run_whisper_parity.sh
#
# Compare large-v3 vs turbo (fetch turbo first via ops/fetch_whisper_model.sh):
#   WHISPER_PATH=/mnt/amc-data/pipeline/models/whisper-large-v3-turbo \
#     bash ops/run_whisper_parity.sh
#
# SAMPLE picks which segments: short (default, N shortest = worst case),
# spread (evenly across durations = representative), random (seeded).
#   SAMPLE=spread LIMIT=80 WHISPER_PATH=... bash ops/run_whisper_parity.sh
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
# Scan ONLY the runs subtree. A bare root/* or root/*/* glob walks the whole data
# root (500K+ input files on Lustre) and hangs for many minutes; runs live under
# amc-runs/<run>/outputs/shard-N, so glob just that (bounded by run count).
runs_root = os.environ.get("AMC_RUNS_ROOT") or os.path.join(root, "amc-runs")
suffix = os.path.join("outputs", "shard-*", ".pii_pipeline", "state", "pipeline.sqlite3")
patterns = [
    os.path.join(runs_root, "*", suffix),
    os.path.join(runs_root, "*", "*", suffix),
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
  echo "FATAL: no run with segments under ${AMC_RUNS_ROOT:-$AMC_IN/amc-runs}. Set AMC_OUT" \
       "or RUN_ROOT+SHARD_INDEX (or AMC_RUNS_ROOT if runs live elsewhere), or run preprocess first." >&2
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

EXTRA_ARGS=()
if [[ -n "${WHISPER_PATH:-}" ]]; then
  if [[ ! -d "$WHISPER_PATH" ]]; then
    echo "FATAL: WHISPER_PATH=$WHISPER_PATH is not a directory (run ops/fetch_whisper_model.sh first)" >&2
    exit 4
  fi
  echo "compare model B (per-segment): $WHISPER_PATH"
  EXTRA_ARGS+=(--compare-path "$WHISPER_PATH")
fi

set +e
LD_LIBRARY_PATH="$ld" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$MAIN_PY" ops/asr_parity_check.py \
    --input "$AMC_IN" --output "$OUT" \
    --model whisper --limit "$LIMIT" --sample "${SAMPLE:-short}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "full log: $LOG"
exit "$rc"
