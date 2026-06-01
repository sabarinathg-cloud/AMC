#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-amc-pipeline:gpu}"
SRC_ROOT="${SRC_ROOT:-/mnt/amc-data}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/amc-data/pipeline/models}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/amc-cache}"
YEAR="${YEAR:-2023}"
CALL_ID="${CALL_ID:-aaf5285a-3d70-56a8-87e3-77c26d547494}"
ONE_IN="${ONE_IN:-/tmp/amc-one-docker}"
ONE_OUT="${ONE_OUT:-/tmp/amc-one-docker-output}"

rm -rf "$ONE_IN" "$ONE_OUT"
mkdir -p "$ONE_IN/$YEAR/$CALL_ID" "$ONE_OUT" "$CACHE_ROOT"
cp "$SRC_ROOT/$YEAR/$CALL_ID/audio.opus" "$ONE_IN/$YEAR/$CALL_ID/audio.opus"

if [[ -n "${AMC_DOCKER_GPU_ARGS+x}" ]]; then
  read -r -a GPU_ARGS <<< "$AMC_DOCKER_GPU_ARGS"
else
  GPU_ARGS=(--gpus all)
fi

run_stage() {
  docker run --rm "${GPU_ARGS[@]}" \
    --ipc=host \
    --shm-size=16g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/cache/huggingface \
    -e TORCH_HOME=/cache/torch \
    -v "$ONE_IN":/data:ro \
    -v "$ONE_OUT":/output \
    -v "$MODEL_ROOT":/models:ro \
    -v "$CACHE_ROOT":/cache \
    "$IMAGE" "$@"
}

run_stage --config /app/docker/config.docker.yaml dry-run --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage preprocess --input /data --output /output --vad-backend silero
run_stage --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models whisper
run_stage --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models qwen --asr-batch-sizes qwen=1
run_stage --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models cohere --asr-batch-sizes cohere=1
run_stage --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models granite --asr-batch-sizes granite=1
run_stage --config /app/docker/config.docker.yaml run-stage normalize --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage consensus --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage pii --input /data --output /output --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
run_stage --config /app/docker/config.docker.yaml run-stage align --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage mask-plan --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage redact --input /data --output /output --mask-strategy beep --allow-fallback-format wav
run_stage --config /app/docker/config.docker.yaml run-stage validate --input /data --output /output
run_stage --config /app/docker/config.docker.yaml run-stage manifest --input /data --output /output

echo "Docker one-call output: $ONE_OUT/$YEAR/$CALL_ID/audio.opus"
