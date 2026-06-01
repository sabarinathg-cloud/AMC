#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-amc-pipeline:gpu}"

docker run --rm --gpus all \
  --entrypoint bash \
  "$IMAGE" \
  -lc 'nvidia-smi && python - <<'"'"'PY'"'"'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY'
