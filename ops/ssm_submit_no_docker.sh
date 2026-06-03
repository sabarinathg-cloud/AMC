#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
AMC_IN="${AMC_IN:?AMC_IN is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
STAGES="${STAGES:-preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest}"

IDS="$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[].InstanceId' \
  --output text | tr '\t' ' ')"

if [[ -z "$IDS" || "$IDS" == "None" ]]; then
  echo "No online SSM instances found for Project=$PROJECT_TAG" >&2
  exit 2
fi

NUM_SHARDS="${NUM_SHARDS:-$(wc -w <<< "$IDS" | tr -d ' ')}"

PARAMS="$(python3 - "$IDS" "$NUM_SHARDS" "$REPO_DIR" "$AMC_IN" "$RUN_ROOT" "$PYTHON_BIN" "$STAGES" <<'PY'
import json, sys
ids, num_shards, repo_dir, amc_in, run_root, python_bin, stages = sys.argv[1:]
commands = [
    "set -e",
    f"export INSTANCE_IDS='{ids}'",
    f"export NUM_SHARDS='{num_shards}'",
    f"export REPO_DIR='{repo_dir}'",
    f"export AMC_IN='{amc_in}'",
    f"export RUN_ROOT='{run_root}'",
    f"export PYTHON_BIN='{python_bin}'",
    f"export STAGES='{stages}'",
    "bash \"$REPO_DIR/ops/run_shard_no_docker.sh\"",
]
print(json.dumps({"commands": commands}))
PY
)"

CMD_ID="$(aws ssm send-command \
  --region "$AWS_REGION" \
  --targets "Key=tag:Project,Values=$PROJECT_TAG" \
  --document-name AWS-RunShellScript \
  --parameters "$PARAMS" \
  --max-concurrency "$NUM_SHARDS" \
  --max-errors "$NUM_SHARDS" \
  --query 'Command.CommandId' \
  --output text)"

echo "$CMD_ID"
