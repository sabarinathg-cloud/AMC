#!/usr/bin/env bash
set -euo pipefail

# Phase A of speaker clustering: fan out the per-shard `speaker_embed` stage to every box
# in the fleet via SSM. Each box embeds its ~1/NUM_SHARDS slice of segments with ReDimNet and
# writes one robust centroid per (call_id, channel) to the SHARED dir $RUN_ROOT/speaker/embed
# as shard-<N>.npz. When all shards finish, run ops/speaker_cluster_once.sh (Phase B).
#
# Usage:
#   AMC_IN=/mnt/amc-data/amc-runs/<run>/input \
#   RUN_ROOT=/mnt/amc-data/amc-runs/<run> \
#   bash ops/speaker_embed_direct.sh
#
# Optional:
#   TORCH_HOME_SHARED=/mnt/amc-data/pipeline/models/torch  # shared ReDimNet hub cache (avoids
#                                                          # 32 concurrent torch.hub downloads)
#   AMC_SPEAKER_MAX_SEGMENTS=30                            # segments embedded per side (0 = all)

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SHARED_ROOT="${AMC_SHARED_ROOT:-/mnt/amc-data}"
AMC_IN="${AMC_IN:?AMC_IN is required (the existing run input dir on the shared mount)}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required (the existing run dir on the shared mount)}"
TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-}"
AMC_SPEAKER_MAX_SEGMENTS="${AMC_SPEAKER_MAX_SEGMENTS:-}"
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

PARAMS="$("$LOCAL_PYTHON" - "$IDS" "$NUM_SHARDS" "$REPO_DIR" "$AMC_IN" "$RUN_ROOT" "$PYTHON_BIN" "$TORCH_HOME_SHARED" "$AMC_SPEAKER_MAX_SEGMENTS" <<'PY'
import json, sys
ids, num_shards, repo_dir, amc_in, run_root, python_bin, torch_home, max_segs = sys.argv[1:]
commands = [
    "set -e",
    f"export INSTANCE_IDS='{ids}'",
    f"export NUM_SHARDS='{num_shards}'",
    f"export REPO_DIR='{repo_dir}'",
    f"export AMC_IN='{amc_in}'",
    f"export RUN_ROOT='{run_root}'",
    f"export PYTHON_BIN='{python_bin}'",
    "export STAGES='speaker_embed'",
    # Embedding reads the per-segment WAVs already under output_root (warm); it does not need the
    # cold S3 input restored, so skip HSM prewarm to start faster.
    "export AMC_PREWARM='0'",
]
if torch_home:
    commands.append(f"export TORCH_HOME='{torch_home}'")
if max_segs:
    commands.append(f"export AMC_SPEAKER_MAX_SEGMENTS='{max_segs}'")
commands.append("mkdir -p \"$RUN_ROOT/logs\" \"$RUN_ROOT/speaker/embed\"")
commands.append(
    "setsid nohup bash \"$REPO_DIR/ops/run_shard_no_docker.sh\" "
    "> \"$RUN_ROOT/logs/launch-speaker-embed-$(hostname -s).log\" 2>&1 < /dev/null & "
    "echo \"detached speaker_embed pid=$! host=$(hostname -s)\""
)
commands.append("sleep 3; echo launch-dispatched")
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

echo "speaker_embed dispatched: CommandId=$CMD_ID shards=$NUM_SHARDS"
echo "Centroids will land in: $RUN_ROOT/speaker/embed/shard-*.npz"
echo "When all $NUM_SHARDS shard-*.npz exist, run: RUN_ROOT=$RUN_ROOT bash ops/speaker_cluster_once.sh"
