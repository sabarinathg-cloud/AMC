#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-amc-pipeline:gpu}"

docker build \
  -f docker/Dockerfile.gpu \
  -t "$IMAGE" \
  .

echo "Built $IMAGE"
