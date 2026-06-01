#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-amc-pipeline:gpu}"
AMC_IN="${AMC_IN:-/mnt/amc-data}"
AMC_OUT="${AMC_OUT:-/mnt/amc-redacted}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/amc-data/pipeline/models}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/amc-cache}"

if [ "$#" -lt 1 ]; then
  echo "usage: docker/run_stage.sh <amc-pipeline args...>" >&2
  echo "example: docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage preprocess --input /data --output /output" >&2
  exit 2
fi

mkdir -p "$AMC_OUT" "$CACHE_ROOT"

if [[ -n "${AMC_DOCKER_GPU_ARGS+x}" ]]; then
  read -r -a GPU_ARGS <<< "$AMC_DOCKER_GPU_ARGS"
else
  GPU_ARGS=(--gpus all)
fi

docker run --rm "${GPU_ARGS[@]}" \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/cache/huggingface \
  -e TORCH_HOME=/cache/torch \
  -v "$AMC_IN":/data:ro \
  -v "$AMC_OUT":/output \
  -v "$MODEL_ROOT":/models:ro \
  -v "$CACHE_ROOT":/cache \
  "$IMAGE" "$@"
