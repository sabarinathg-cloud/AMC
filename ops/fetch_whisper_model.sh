#!/usr/bin/env bash
# Stage a faster-whisper (CTranslate2) model into the shared model root so the
# WhisperAdapter can load it with local_files_only=True.
#
# Default target is large-v3-turbo: ~5-6x faster decode than large-v3 with WER
# within ~1% on most benchmarks (the best-WER fast variant -- distil-large-v3 is
# faster still but noticeably worse and English-only, so we do not default to it).
# large-v3 stays the pipeline default; turbo is opt-in until you A/B the WER.
#
# Usage:
#   bash ops/fetch_whisper_model.sh                       # turbo -> $MODEL_ROOT/whisper-large-v3-turbo
#   REPO_ID=Systran/faster-whisper-large-v3 DEST_NAME=whisper-large-v3 bash ops/fetch_whisper_model.sh
#
# Env:
#   MODEL_ROOT  shared model dir       (default /mnt/amc-data/pipeline/models)
#   REPO_ID     HF repo (CT2 format)   (default deepdml/faster-whisper-large-v3-turbo-ct2)
#   DEST_NAME   subdir under MODEL_ROOT (default whisper-large-v3-turbo)
#   MAIN_PY     python with huggingface_hub (default: main venv, then python3)
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/amc-data/pipeline/models}"
REPO_ID="${REPO_ID:-deepdml/faster-whisper-large-v3-turbo-ct2}"
DEST_NAME="${DEST_NAME:-whisper-large-v3-turbo}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"

if [ -z "${MAIN_PY:-}" ]; then
  if [ -x "$VENV_ROOT/main/bin/python" ]; then
    MAIN_PY="$VENV_ROOT/main/bin/python"
  else
    MAIN_PY="$(command -v python3 || command -v python)"
  fi
fi

DEST="$MODEL_ROOT/$DEST_NAME"
echo "fetch_whisper_model: repo=$REPO_ID -> $DEST (python=$MAIN_PY)"
mkdir -p "$DEST"

"$MAIN_PY" - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
# CT2 models are small flat dirs (model.bin/config.json/tokenizer.json/vocabulary).
path = snapshot_download(
    repo_id=repo_id,
    local_dir=dest,
    local_dir_use_symlinks=False,
    allow_patterns=["*.bin", "*.json", "*.txt", "vocabulary*", "tokenizer*", "preprocessor*"],
)
print(f"  downloaded to {path}")
PY

# Sanity: faster-whisper must be able to open it locally.
"$MAIN_PY" - "$DEST" <<'PY'
import sys
from faster_whisper import WhisperModel

dest = sys.argv[1]
WhisperModel(dest, device="cpu", compute_type="int8", local_files_only=True)
print(f"  OK: faster-whisper loaded {dest}")
PY

echo "fetch_whisper_model: done."
echo "Point the pipeline at it via config asr_models.whisper.path = $DEST"
echo "(then A/B WER before making it the default)."
