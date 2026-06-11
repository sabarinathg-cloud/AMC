#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"

if [[ -n "${1:-}" ]]; then
  aws ssm list-command-invocations \
    --region "$AWS_REGION" \
    --command-id "$1" \
    --details \
    --query 'CommandInvocations[].{Instance:InstanceId,Status:Status,Response:StatusDetails}' \
    --output table
fi

if [[ -n "${RUN_ROOT:-}" ]]; then
  # Pick ONE online instance. Use a client-side awk filter rather than a JMESPath
  # `[0].InstanceId` query: describe-instance-information paginates (10/page), and the
  # AWS CLI applies `--query` PER PAGE, so `[0]` returns one id *per page* (e.g. 4 ids
  # for a 32-box fleet) -- which then fails send-command's single-id validation.
  INSTANCE_ID="$(aws ssm describe-instance-information \
    --region "$AWS_REGION" \
    --filters "Key=tag:Project,Values=$PROJECT_TAG" \
    --query 'InstanceInformationList[].[InstanceId,PingStatus]' \
    --output text | awk '$2=="Online"{print $1; exit}')"
  if [[ -z "$INSTANCE_ID" ]]; then
    echo "No online SSM instances found for Project=$PROJECT_TAG" >&2
    exit 2
  fi
  STATUS_CMD_ID="$(aws ssm send-command \
    --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript \
    --parameters "$(python3 - "$RUN_ROOT" <<'PY'
import json, sys
run_root = sys.argv[1]
print(json.dumps({"commands": [f"python3.10 /mnt/amc-data/AMC/ops/print_run_status.py --run-root {run_root}"]}))
PY
)" \
    --query 'Command.CommandId' \
    --output text)"
  # print_run_status.py scans every shard's state on shared storage; a fixed 2s sleep
  # often catches it mid-run and returns empty output. Poll until the invocation settles.
  for _ in $(seq 1 30); do
    ST="$(aws ssm get-command-invocation --region "$AWS_REGION" \
      --command-id "$STATUS_CMD_ID" --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text 2>/dev/null || echo Pending)"
    [[ "$ST" == "Success" || "$ST" == "Failed" || "$ST" == "Cancelled" || "$ST" == "TimedOut" ]] && break
    sleep 3
  done
  aws ssm get-command-invocation \
    --region "$AWS_REGION" \
    --command-id "$STATUS_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text
fi
