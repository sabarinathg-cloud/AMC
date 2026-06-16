#!/usr/bin/env bash
set -euo pipefail

# All paths in this script (e.g. /mnt/amc-data/...) are REMOTE Linux paths. On Git Bash /
# MSYS (Windows) these would be mangled into Windows paths (C:/Program Files/Git/mnt/...)
# when passed as arguments to a native program like python.exe. Disable that conversion.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
# Local Python used ONLY to build the SSM JSON parameters on THIS machine (not the fleet).
# Prefer python3, fall back to python (e.g. Git Bash on Windows). Override with LOCAL_PYTHON.
LOCAL_PYTHON="${LOCAL_PYTHON:-}"
if [[ -z "$LOCAL_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    LOCAL_PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    LOCAL_PYTHON="python"
  else
    echo "No local Python found (need python3 or python) to build SSM parameters." >&2
    exit 2
  fi
fi

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
    --parameters "$("$LOCAL_PYTHON" - "$RUN_ROOT" <<'PY'
import json, sys
run_root = sys.argv[1]
print(json.dumps({"commands": [f"python3.10 /mnt/amc-data/AMC/ops/print_run_status.py --run-root {run_root}"]}))
PY
)" \
    --query 'Command.CommandId' \
    --output text)"
  # print_run_status.py scans every shard's state on shared storage; under load (32 workers
  # writing while we read 32 SQLite DBs over Lustre) it can take minutes, so poll generously.
  # Override the wait with STATUS_POLL_TRIES (x3s); default 200 tries = 10 minutes.
  ST="Pending"
  for _ in $(seq 1 "${STATUS_POLL_TRIES:-200}"); do
    ST="$(aws ssm get-command-invocation --region "$AWS_REGION" \
      --command-id "$STATUS_CMD_ID" --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text 2>/dev/null || echo Pending)"
    [[ "$ST" == "Success" || "$ST" == "Failed" || "$ST" == "Cancelled" || "$ST" == "TimedOut" ]] && break
    sleep 3
  done
  if [[ "$ST" != "Success" ]]; then
    echo "status command did not finish (last state: $ST). Re-run, or inspect command-id $STATUS_CMD_ID directly." >&2
  fi
  OUT="$(aws ssm get-command-invocation \
    --region "$AWS_REGION" \
    --command-id "$STATUS_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text)"
  if [[ -z "${OUT// }" ]]; then
    echo "(no status output yet; command state=$ST, command-id=$STATUS_CMD_ID)" >&2
  else
    printf '%s\n' "$OUT"
  fi
fi
