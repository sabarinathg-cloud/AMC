#!/usr/bin/env bash
# Stage the Parakeet TDT weights into the shared model root so ParakeetAdapter can
# load them with local_files_only=True.
#
# Default target is parakeet-tdt-0.6b-v3 (25 languages, includes en + es), the
# Whisper replacement candidate: 6.32 avg WER vs Whisper large-v3's 7.44 on the
# Open ASR Leaderboard, at ~23x the throughput. Use v2 instead if the corpus turns
# out to be overwhelmingly English -- it is English-only but scores 6.05.
#
# Run this ONCE from a single instance: the model root is on the shared mount, so
# every box reads the same copy (same reason ops/fetch_redimnet_model.sh exists).
#
# Usage:
#   bash ops/fetch_parakeet_model.sh                      # v3 -> $MODEL_ROOT/parakeet-tdt-0.6b-v3
#   REPO_ID=nvidia/parakeet-tdt-0.6b-v2 DEST_NAME=parakeet-tdt-0.6b-v2 bash ops/fetch_parakeet_model.sh
#
# Env:
#   MODEL_ROOT   shared model dir        (default /mnt/amc-data/pipeline/models)
#   REPO_ID      HF repo                 (default nvidia/parakeet-tdt-0.6b-v3)
#   DEST_NAME    subdir under MODEL_ROOT (default parakeet-tdt-0.6b-v3)
#   PARAKEET_PY  python with transformers>=5.9.0 (default: parakeet venv, then python3)
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/amc-data/pipeline/models}"
REPO_ID="${REPO_ID:-nvidia/parakeet-tdt-0.6b-v3}"
DEST_NAME="${DEST_NAME:-parakeet-tdt-0.6b-v3}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"

if [ -z "${PARAKEET_PY:-}" ]; then
  if [ -x "$VENV_ROOT/parakeet/bin/python" ]; then
    PARAKEET_PY="$VENV_ROOT/parakeet/bin/python"
  else
    PARAKEET_PY="$(command -v python3 || command -v python)"
  fi
fi

DEST="$MODEL_ROOT/$DEST_NAME"
echo "fetch_parakeet_model: repo=$REPO_ID -> $DEST (python=$PARAKEET_PY)"
mkdir -p "$DEST"

"$PARAKEET_PY" - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
# Weights + processor only. The repo also ships a .nemo checkpoint (~2.4 GB) that
# duplicates the safetensors and is only needed by the NeMo runtime, which we do
# not use -- excluding it roughly halves the download.
path = snapshot_download(
    repo_id=repo_id,
    local_dir=dest,
    allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
    ignore_patterns=["*.nemo", "*.ckpt", "*.pt"],
)
print(f"  downloaded to {path}")
PY

# Sanity: the adapter's exact load path must work offline, on CPU.
"$PARAKEET_PY" - "$DEST" <<'PY'
import sys

import torch
import transformers

# transformers 5.x resolves FP8 MX dtypes at import; the venv is on torch 2.5.1,
# which predates them. Same shim the cohere/parakeet adapters apply at load.
from amc_pipeline.transcription import _ensure_torch_fp8_dtype_shim

_ensure_torch_fp8_dtype_shim(torch)

try:
    from transformers import AutoModelForTDT, AutoProcessor, ParakeetForTDT  # noqa: F401
except Exception as exc:
    raise SystemExit(
        f"transformers=={transformers.__version__} / torch=={torch.__version__} cannot import "
        f"ParakeetForTDT ({type(exc).__name__}: {exc}). Needs transformers>=5.9.0; re-run with "
        "PARAKEET_PY pointing at the parakeet venv."
    ) from exc

dest = sys.argv[1]
processor = AutoProcessor.from_pretrained(dest, local_files_only=True)
model = AutoModelForTDT.from_pretrained(dest, local_files_only=True)
params = sum(p.numel() for p in model.parameters())
print(f"  OK: loaded {dest}")
print(f"      params={params / 1e6:.0f}M  sample_rate={processor.feature_extractor.sampling_rate}")
PY

echo "fetch_parakeet_model: done."
echo "Benchmark it before switching:"
echo "  $PARAKEET_PY ops/asr_model_bench.py --run-root /mnt/amc-data/amc-runs/2022-full \\"
echo "      --model parakeet --model-path $DEST --limit 2000"
