#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-amc-pipeline:gpu}"

if [[ -n "${AMC_DOCKER_GPU_ARGS+x}" ]]; then
  read -r -a GPU_ARGS <<< "$AMC_DOCKER_GPU_ARGS"
else
  GPU_ARGS=(--gpus all)
fi

docker run --rm "${GPU_ARGS[@]}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
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
