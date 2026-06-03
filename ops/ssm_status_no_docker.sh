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
  INSTANCE_ID="$(aws ssm describe-instance-information \
    --region "$AWS_REGION" \
    --filters "Key=tag:Project,Values=$PROJECT_TAG" \
    --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[0].InstanceId' \
    --output text)"
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
  sleep 2
  aws ssm get-command-invocation \
    --region "$AWS_REGION" \
    --command-id "$STATUS_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --query '{Status:Status,Output:StandardOutputContent,Error:StandardErrorContent}' \
    --output json
fi
