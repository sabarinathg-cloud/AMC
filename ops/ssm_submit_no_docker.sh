#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
# SHARED_ROOT must be the cluster-wide shared mount. On this fleet that is the FSx for Lustre
# mount at /mnt/amc-data; /mnt/amc-runs is a LOCAL directory on each instance and must NOT be
# used for run state (only the submitting host would have the input, so every other worker
# discovers an empty input and no-ops).
SHARED_ROOT="${AMC_SHARED_ROOT:-/mnt/amc-data}"
AMC_IN="${AMC_IN:?AMC_IN is required (must be on the shared mount, e.g. $SHARED_ROOT/amc-runs/<run>/input)}"
RUN_ROOT="${RUN_ROOT:-$SHARED_ROOT/amc-runs/run-$(date -u +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
STAGES="${STAGES:-preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest}"
# Optional fleet-wide ASR batch override (from ops/asr_batch_sweep.py). Forwarded to every shard
# as --asr-batch-sizes. Empty by default -> each model uses its config default.
AMC_ASR_BATCH_SIZES="${AMC_ASR_BATCH_SIZES:-}"
# Optional VAD device override forwarded to every shard. Empty -> code default (CPU; see
# amc_pipeline/segmentation._resolve_vad_device). Set to cuda/cuda:0 to force GPU VAD.
AMC_VAD_DEVICE="${AMC_VAD_DEVICE:-}"

# Guard: AMC_IN and RUN_ROOT MUST live on the shared mount, or non-shard-0 workers silently
# get an empty input. Override only with eyes open via AMC_ALLOW_NONSHARED=1.
assert_shared() {  # $1=label  $2=path
  case "$2" in
    "$SHARED_ROOT"/*) : ;;
    *)
      echo "ERROR: $1=$2 is not under the shared mount $SHARED_ROOT." >&2
      echo "       /mnt/amc-runs is LOCAL per instance on this fleet -- workers would see an empty input." >&2
      echo "       Use a path under $SHARED_ROOT (e.g. $SHARED_ROOT/amc-runs/...), or set AMC_ALLOW_NONSHARED=1 to override." >&2
      [[ "${AMC_ALLOW_NONSHARED:-0}" == "1" ]] || exit 2
      ;;
  esac
}
assert_shared AMC_IN "$AMC_IN"
assert_shared RUN_ROOT "$RUN_ROOT"

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

PARAMS="$(python3 - "$IDS" "$NUM_SHARDS" "$REPO_DIR" "$AMC_IN" "$RUN_ROOT" "$PYTHON_BIN" "$STAGES" "$AMC_ASR_BATCH_SIZES" "$AMC_VAD_DEVICE" <<'PY'
import json, sys
ids, num_shards, repo_dir, amc_in, run_root, python_bin, stages, asr_batch, vad_device = sys.argv[1:]
commands = [
    "set -e",
    f"export INSTANCE_IDS='{ids}'",
    f"export NUM_SHARDS='{num_shards}'",
    f"export REPO_DIR='{repo_dir}'",
    f"export AMC_IN='{amc_in}'",
    f"export RUN_ROOT='{run_root}'",
    f"export PYTHON_BIN='{python_bin}'",
    f"export STAGES='{stages}'",
]
if asr_batch:
    commands.append(f"export AMC_ASR_BATCH_SIZES='{asr_batch}'")
if vad_device:
    commands.append(f"export AMC_VAD_DEVICE='{vad_device}'")
# Detach the shard worker from the SSM command's lifetime. An SSM RunShellScript
# invocation is bounded (max executionTimeout 48h) and SIGKILLs its whole process
# group on timeout/completion -- which previously killed multi-day runs ~minutes in
# (ResponseCode 137). setsid+nohup move the worker into its own session so it keeps
# running after this command returns; we just kick it off and exit immediately. The
# worker's own logs live under $RUN_ROOT/logs/shard-*; this launch log captures any
# startup error before the worker takes over logging.
commands.append("mkdir -p \"$RUN_ROOT/logs\"")
commands.append(
    "setsid nohup bash \"$REPO_DIR/ops/run_shard_no_docker.sh\" "
    "> \"$RUN_ROOT/logs/launch-$(hostname -s).log\" 2>&1 < /dev/null & "
    "echo \"detached run_shard pid=$! host=$(hostname -s)\""
)
# Give the worker a moment to fork/exec so a launch crash surfaces in this invocation.
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

echo "$CMD_ID"
