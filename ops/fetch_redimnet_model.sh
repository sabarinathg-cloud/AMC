#!/usr/bin/env bash
# Pre-stage the ReDimNet speaker encoder (torch.hub "IDRnD/ReDimNet") into a SHARED
# torch.hub cache so the 32 boxes running `speaker_embed` reuse it instead of each
# hitting GitHub + the checkpoint host concurrently (which rate-limits/fails at boot).
#
# It clones the hub repo and downloads the ReDimNet-M (ft_mix / vb2+vox2+cnc) checkpoint
# into $TORCH_HOME_SHARED/hub. Point the embed stage at the same dir by exporting
# TORCH_HOME to it -- ops/speaker_embed_direct.sh forwards TORCH_HOME_SHARED for you.
#
# Usage:
#   bash ops/fetch_redimnet_model.sh
#   TORCH_HOME_SHARED=/mnt/amc-data/pipeline/models/torch bash ops/fetch_redimnet_model.sh
#
# Env:
#   TORCH_HOME_SHARED  shared torch hub cache (default /mnt/amc-data/pipeline/models/torch)
#   MODEL_NAME/TRAIN_TYPE/DATASET  ReDimNet variant (defaults match the pipeline: M/ft_mix/vb2+vox2+cnc)
#   MAIN_PY            python with torch/torchaudio (default: main venv, then python3)
set -euo pipefail

TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-/mnt/amc-data/pipeline/models/torch}"
MODEL_NAME="${MODEL_NAME:-M}"
TRAIN_TYPE="${TRAIN_TYPE:-ft_mix}"
DATASET="${DATASET:-vb2+vox2+cnc}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"

if [ -z "${MAIN_PY:-}" ]; then
  if [ -x "$VENV_ROOT/main/bin/python" ]; then
    MAIN_PY="$VENV_ROOT/main/bin/python"
  else
    MAIN_PY="$(command -v python3 || command -v python)"
  fi
fi

export TORCH_HOME="$TORCH_HOME_SHARED"
mkdir -p "$TORCH_HOME"
echo "fetch_redimnet_model: TORCH_HOME=$TORCH_HOME variant=$MODEL_NAME/$TRAIN_TYPE/$DATASET (python=$MAIN_PY)"

"$MAIN_PY" - "$MODEL_NAME" "$TRAIN_TYPE" "$DATASET" <<'PY'
import sys
import torch

model_name, train_type, dataset = sys.argv[1:]
# Downloads + caches the hub repo and checkpoint under TORCH_HOME/hub. A CPU load is enough
# to populate the cache; the fleet then loads from disk with no network.
model = torch.hub.load(
    "IDRnD/ReDimNet",
    "ReDimNet",
    model_name=model_name,
    train_type=train_type,
    dataset=dataset,
    trust_repo=True,
)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  OK: ReDimNet-{model_name}/{train_type}/{dataset} loaded, params={n_params/1e6:.2f}M")
PY

echo "fetch_redimnet_model: done. Shared cache at $TORCH_HOME/hub"
echo "Run embed with: TORCH_HOME_SHARED=$TORCH_HOME_SHARED ... bash ops/speaker_embed_direct.sh"
