#!/usr/bin/env bash
# Run a script on one instance via SSM and print its output.
#
# The script body is read from stdin and shipped base64-encoded, because SSM
# passes commands through a JSON document and then /bin/sh: quotes, parentheses
# and newlines in a normal bash script get mangled on the way.
#
# Usage: ssm_run.sh <instance-id> [timeout-seconds] < script.sh
set -Eeuo pipefail

INSTANCE_ID="${1:?instance id required}"
TIMEOUT="${2:-3600}"
POLL_SECONDS="${SSM_POLL_SECONDS:-10}"

payload="$(base64 | tr -d '\n')"

command_id="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --timeout-seconds "$TIMEOUT" \
  --parameters "commands=[\"echo $payload | base64 -d | bash\"]" \
  --query 'Command.CommandId' --output text)"

echo "command-id: $command_id" >&2

deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  sleep "$POLL_SECONDS"
  status="$(aws ssm get-command-invocation \
    --command-id "$command_id" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "$status" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
  if (( $(date +%s) > deadline )); then
    echo "status: timed out waiting (last=$status)" >&2
    break
  fi
done

aws ssm get-command-invocation \
  --command-id "$command_id" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text
err="$(aws ssm get-command-invocation \
  --command-id "$command_id" --instance-id "$INSTANCE_ID" \
  --query 'StandardErrorContent' --output text)"
[[ -z "$err" || "$err" == "None" ]] || { echo "--- stderr ---"; echo "$err"; }
echo "--- status: $status ---"
