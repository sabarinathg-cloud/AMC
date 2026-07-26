#!/usr/bin/env bash
# Run a local bash script on a fleet instance via SSM and stream back the result.
#
# SSM's AWS-RunShellScript executes its payload under /bin/sh (dash on Ubuntu), which
# chokes on bash syntax. So we base64 the script locally and decode|bash it remotely --
# the only thing SSM parses is a trivially POSIX-safe one-liner.
#
# Usage: ssmrun.sh <script-file> [instance-id] [timeout-seconds]
set -euo pipefail

SCRIPT="${1:?usage: ssmrun.sh <script-file> [instance-id] [timeout-seconds]}"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
TIMEOUT="${3:-3600}"

TARGET="${2:-}"
if [[ -z "$TARGET" ]]; then
  TARGET="$(aws ssm describe-instance-information --region "$AWS_REGION" \
    --filters "Key=tag:Project,Values=$PROJECT_TAG" \
    --query 'InstanceInformationList[].[InstanceId,PingStatus]' \
    --output text | awk '$2=="Online"{print $1}' | sort | head -1)"
fi
[[ -n "$TARGET" && "$TARGET" != "None" ]] || { echo "no online instance" >&2; exit 2; }

PAYLOAD="$(base64 < "$SCRIPT" | tr -d '\n')"
CID="$(aws ssm send-command --region "$AWS_REGION" \
  --instance-ids "$TARGET" --document-name AWS-RunShellScript \
  --timeout-seconds "$TIMEOUT" \
  --parameters "commands=[\"echo $PAYLOAD | base64 -d | bash\"]" \
  --query 'Command.CommandId' --output text)"

echo "[ssmrun] instance=$TARGET command=$CID script=$SCRIPT" >&2
while :; do
  sleep 10
  STATUS="$(aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$CID" --instance-id "$TARGET" --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "$STATUS" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
done

echo "[ssmrun] status=$STATUS" >&2
aws ssm get-command-invocation --region "$AWS_REGION" \
  --command-id "$CID" --instance-id "$TARGET" \
  --query 'StandardOutputContent' --output text
ERR="$(aws ssm get-command-invocation --region "$AWS_REGION" \
  --command-id "$CID" --instance-id "$TARGET" --query 'StandardErrorContent' --output text)"
[[ -n "$ERR" && "$ERR" != "None" ]] && { echo "--- stderr ---" >&2; echo "$ERR" >&2; }
[[ "$STATUS" == "Success" ]]
