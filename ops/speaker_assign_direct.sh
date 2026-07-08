#!/usr/bin/env bash
set -euo pipefail

# Phase C of speaker clustering: fan out the per-shard `speaker_assign` stage to every box via
# SSM. Each box reads the global $RUN_ROOT/speaker/clusters.parquet (from Phase B,
# ops/speaker_cluster_once.sh), joins speaker_cluster_id onto every segment, and regenerates
# all_segments.parquet with the new column.
#
# Usage:
#   AMC_IN=/mnt/amc-data/amc-runs/<run>/input \
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> \
#   bash ops/speaker_assign_direct.sh

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SHARED_ROOT="${AMC_SHARED_ROOT:-/mnt/amc-data}"
AMC_IN="${AMC_IN:?AMC_IN is required (the existing run input dir on the shared mount)}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required (the existing run dir on the shared mount)}"
AMC_MANIFEST_LEAN="${AMC_MANIFEST_LEAN:-1}"
LOCAL_PYTHON="${LOCAL_PYTHON:-}"
if [[ -z "$LOCAL_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then LOCAL_PYTHON="python3";
  elif command -v python >/dev/null 2>&1; then LOCAL_PYTHON="python";
  else echo "No local Python found (need python3 or python)." >&2; exit 2; fi
fi

case "$AMC_IN"   in "$SHARED_ROOT"/*) : ;; *) echo "ERROR: AMC_IN must be under $SHARED_ROOT" >&2; exit 2 ;; esac
case "$RUN_ROOT" in "$SHARED_ROOT"/*) : ;; *) echo "ERROR: RUN_ROOT must be under $SHARED_ROOT" >&2; exit 2 ;; esac

IDS="$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[].InstanceId' \
  --output text | tr '\t' ' ')"
if [[ -z "$IDS" || "$IDS" == "None" ]]; then
  echo "No online SSM instances found for Project=$PROJECT_TAG" >&2; exit 2
fi
NUM_SHARDS="${NUM_SHARDS:-$(wc -w <<< "$IDS" | tr -d ' ')}"

PARAMS="$("$LOCAL_PYTHON" - "$IDS" "$NUM_SHARDS" "$REPO_DIR" "$AMC_IN" "$RUN_ROOT" "$PYTHON_BIN" "$AMC_MANIFEST_LEAN" <<'PY'
import json, sys
ids, num_shards, repo_dir, amc_in, run_root, python_bin, lean = sys.argv[1:]
commands = [
    "set -e",
    f"export INSTANCE_IDS='{ids}'",
    f"export NUM_SHARDS='{num_shards}'",
    f"export REPO_DIR='{repo_dir}'",
    f"export AMC_IN='{amc_in}'",
    f"export RUN_ROOT='{run_root}'",
    f"export PYTHON_BIN='{python_bin}'",
    "export STAGES='speaker_assign'",
    f"export AMC_MANIFEST_LEAN='{lean}'",
    "export AMC_PREWARM='0'",
    # Fail loudly per box if Phase B has not produced the global mapping yet.
    "test -f \"$RUN_ROOT/speaker/clusters.parquet\" || test -f \"$RUN_ROOT/speaker/clusters.jsonl\" "
    "|| { echo \"MISSING $RUN_ROOT/speaker/clusters.parquet -- run ops/speaker_cluster_once.sh first\" >&2; exit 4; }",
    "mkdir -p \"$RUN_ROOT/logs\"",
    "setsid nohup bash \"$REPO_DIR/ops/run_shard_no_docker.sh\" "
    "> \"$RUN_ROOT/logs/launch-speaker-assign-$(hostname -s).log\" 2>&1 < /dev/null & "
    "echo \"detached speaker_assign pid=$! host=$(hostname -s)\"",
    "sleep 3; echo launch-dispatched",
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

echo "speaker_assign dispatched: CommandId=$CMD_ID shards=$NUM_SHARDS"
echo "Each shard's all_segments.parquet gains a speaker_cluster_id column."
