#!/usr/bin/env bash
set -euo pipefail

# Phase B of speaker clustering (RUN ONCE, on a single box): gather every shard's
# per-(call_id, channel) centroids from $RUN_ROOT/speaker/embed/shard-*.npz and run the global
# strict constrained-union clustering, producing $RUN_ROOT/speaker/clusters.parquet
# (call_id, channel -> speaker_cluster_id). This MUST be global -- the same speaker appears in
# calls on different shards, so a per-shard clustering would leak speakers across train/test.
#
# Runs the cluster-speakers CLI on ONE fleet instance via SSM (it has the venv + faiss/sklearn +
# pyarrow and the shared mount), waits for it, and prints the summary. Set NUM_SHARDS to require
# a specific count of shard-*.npz before clustering (default: cluster whatever is present).
#
# Usage:
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> [NUM_SHARDS=32] bash ops/speaker_cluster_once.sh

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
SHARED_ROOT="${AMC_SHARED_ROOT:-/mnt/amc-data}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required (the existing run dir on the shared mount)}"
EMBED_ROOT="$RUN_ROOT/speaker/embed"
OUT_PATH="$RUN_ROOT/speaker/clusters.parquet"
WAIT_SECS="${WAIT_SECS:-1800}"        # how long to wait for all shard-*.npz to appear
EXPECT_SHARDS="${NUM_SHARDS:-0}"       # 0 = don't require a specific count

case "$RUN_ROOT" in "$SHARED_ROOT"/*) : ;; *) echo "ERROR: RUN_ROOT must be under $SHARED_ROOT" >&2; exit 2 ;; esac

# NOTE: pick the first Online instance via awk on (InstanceId, PingStatus) rather than a
# backtick JMESPath filter -- `[?PingStatus==`Online`][0]` proved unreliable across AWS CLI
# versions (returned multiple ids), and SSM rejects a multi-id --instance-ids scalar.
TARGET="$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'InstanceInformationList[].[InstanceId,PingStatus]' \
  --output text | awk '$2=="Online"{print $1}' | sort | head -1)"
if [[ -z "$TARGET" || "$TARGET" == "None" ]]; then
  echo "No online SSM instance found for Project=$PROJECT_TAG" >&2; exit 2
fi
echo "Clustering on instance: $TARGET"
echo "Reading centroids from:  $EMBED_ROOT"
echo "Writing mapping to:      $OUT_PATH"

# SSM AWS-RunShellScript executes the payload under /bin/sh (dash on Ubuntu), which chokes on
# the bash-style body below. So we base64-encode the script locally and run it under bash
# remotely -- the two commands we hand SSM are trivially POSIX-safe.
REMOTE_SCRIPT="$(cat <<REMOTE
set -e
MAIN_PY="$VENV_ROOT/main/bin/python"
[ -x "\$MAIN_PY" ] || MAIN_PY="\$(command -v python3.10 || command -v python3)"
mkdir -p "$RUN_ROOT/speaker" "$RUN_ROOT/logs"
# Wait for all shard centroids (bounded).
deadline=\$(( \$(date +%s) + $WAIT_SECS ))
while :; do
  have=\$(ls -1 "$EMBED_ROOT"/shard-*.npz 2>/dev/null | wc -l | tr -d ' ')
  if [ "$EXPECT_SHARDS" -gt 0 ]; then
    [ "\$have" -ge "$EXPECT_SHARDS" ] && break
  else
    [ "\$have" -gt 0 ] && break
  fi
  if [ \$(date +%s) -ge \$deadline ]; then
    echo "TIMEOUT waiting for shard-*.npz (have=\$have expect=$EXPECT_SHARDS)" >&2
    [ "\$have" -gt 0 ] || exit 5
    echo "proceeding with \$have shard(s)"
    break
  fi
  sleep 15
done
cd "$REPO_DIR"
"\$MAIN_PY" -m amc_pipeline.cli cluster-speakers --embed-root "$EMBED_ROOT" --out "$OUT_PATH" \
  2>&1 | tee "$RUN_ROOT/logs/speaker_cluster_once.log"
REMOTE
)"

PARAMS="$(printf '%s' "$REMOTE_SCRIPT" | "${LOCAL_PYTHON:-python3}" -c 'import json,sys,base64
b64 = base64.b64encode(sys.stdin.buffer.read()).decode()
cmds = ["set -e", "printf %s " + json.dumps(b64) + " | base64 -d | bash"]
print(json.dumps({"commands": cmds}))')"

CMD_ID="$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "$TARGET" \
  --document-name AWS-RunShellScript \
  --comment "speaker cluster-speakers (global)" \
  --parameters "$PARAMS" \
  --query 'Command.CommandId' \
  --output text)"

echo "cluster-speakers dispatched: CommandId=$CMD_ID; polling..."
while :; do
  sleep 15
  STATUS="$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" --instance-id "$TARGET" --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "$STATUS" in
    Success)
      echo "----- cluster-speakers output -----"
      aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" --instance-id "$TARGET" --query 'StandardOutputContent' --output text
      echo "clusters.parquet ready: $OUT_PATH"
      echo "Next: AMC_IN=... RUN_ROOT=$RUN_ROOT bash ops/speaker_assign_direct.sh"
      exit 0
      ;;
    Failed|Cancelled|TimedOut)
      echo "cluster-speakers $STATUS. stderr:" >&2
      aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" --instance-id "$TARGET" --query 'StandardErrorContent' --output text >&2
      exit 1
      ;;
    *) echo "status=$STATUS ..." ;;
  esac
done
