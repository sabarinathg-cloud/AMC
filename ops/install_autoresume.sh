#!/usr/bin/env bash
# Install fleet-wide auto-resume for the AMC pipeline.
#
# What it does:
#   1. Builds a durable run config (run.env) from the env vars below.
#   2. On ONE instance: updates the shared repo checkout and writes the run config to
#      $RUN_ROOT/run.env and the stable pointer $AMC_RUN_ENV (default
#      /mnt/amc-runs/active.env). (Single writer avoids the NFS ref/race problems.)
#   3. On ALL online instances: installs ops/resume_shard.sh to /usr/local/bin and the
#      systemd unit, then `systemctl enable --now amc-shard.service`.
#
# After this, each instance runs its shard to completion AND auto-resumes on reboot /
# spot reclaim / crash, with no further SSM submits. To stop the fleet:
#   touch $RUN_ROOT/STOP   (and optionally `systemctl disable --now amc-shard.service`).
#
# Required env: AMC_IN, RUN_ROOT
# Optional env: AWS_REGION, PROJECT_TAG, REPO_DIR, PYTHON_BIN, HASH_MODE, NUM_SHARDS,
#               STAGES, AUTO_PULL, AMC_RUN_ENV
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
HASH_MODE="${HASH_MODE:-path}"
AMC_IN="${AMC_IN:?AMC_IN is required (input root the pipeline discovers)}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required, e.g. /mnt/amc-runs/2026-smoke20}"
STAGES="${STAGES:-preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest}"
AUTO_PULL="${AUTO_PULL:-0}"
AMC_RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-runs/active.env}"

IDS="$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[].InstanceId' \
  --output text | tr '\t' ' ')"

if [[ -z "$IDS" || "$IDS" == "None" ]]; then
  echo "No online SSM instances found for Project=$PROJECT_TAG" >&2
  exit 2
fi
ONLINE_COUNT="$(wc -w <<< "$IDS" | tr -d ' ')"
FIRST_ID="$(awk '{print $1}' <<< "$IDS")"

# Shard count is the fixed data-partition count and the CONCURRENCY CAP: at most
# NUM_SHARDS machines can run at once. It is baked into the call->shard hash and the
# on-disk state, so it CANNOT change mid-run. Default it well above any fleet size so the
# run stays elastic -- add machines anytime and they immediately claim unworked shards,
# with the rest idle only if the fleet ever exceeds NUM_SHARDS. Override by exporting
# NUM_SHARDS explicitly.
DEFAULT_SHARDS="${AMC_DEFAULT_SHARDS:-256}"
if [[ -z "${NUM_SHARDS:-}" ]]; then
  if (( ONLINE_COUNT > DEFAULT_SHARDS )); then
    NUM_SHARDS="$ONLINE_COUNT"
  else
    NUM_SHARDS="$DEFAULT_SHARDS"
  fi
fi

echo "Instances : $IDS ($ONLINE_COUNT online)"
echo "Shards    : $NUM_SHARDS  (concurrency cap; >= fleet size keeps the run elastic)"
echo "AMC_IN    : $AMC_IN"
echo "RUN_ROOT  : $RUN_ROOT"
echo "Run config: $AMC_RUN_ENV"

# Best-effort sanity check: warn if shards would be so small that per-shard model reloads
# dominate. calls/shard below ~1000 is wasteful. This is approximate (counts immediate
# subdirectories of AMC_IN as a proxy for call count) and never blocks the install.
# Override the estimate with EXPECTED_CALLS, or skip entirely with SKIP_SHARD_CHECK=1.
if [[ "${SKIP_SHARD_CHECK:-0}" != "1" ]]; then
  MIN_PER_SHARD="${AMC_MIN_CALLS_PER_SHARD:-1000}"
  if [[ -n "${EXPECTED_CALLS:-}" ]]; then
    CALL_EST="$EXPECTED_CALLS"; CAPPED=""
  else
    # Bounded scan with early exit at NUM_SHARDS*MIN_PER_SHARD: confirms "healthy" fast
    # even on a 500K-call directory without a full enumeration.
    read -r CALL_EST CAPPED < <(python3 - "$AMC_IN" "$NUM_SHARDS" "$MIN_PER_SHARD" <<'PY'
import os, sys
amc_in, num_shards, min_per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cap = num_shards * min_per
n = 0; capped = 0
try:
    with os.scandir(amc_in) as it:
        for e in it:
            if e.name.startswith('.'):
                continue
            try:
                if e.is_dir():
                    n += 1
                    if n >= cap:
                        capped = 1
                        break
            except OSError:
                pass
except (FileNotFoundError, NotADirectoryError):
    print("unknown 0"); sys.exit(0)
print(f"{n} {capped}")
PY
)
  fi
  if [[ "${CAPPED:-0}" == "1" ]]; then
    echo "Shard size: OK (>= $MIN_PER_SHARD calls/shard)"
  elif [[ "$CALL_EST" == "unknown" || -z "$CALL_EST" ]]; then
    echo "Shard size: could not estimate call count from $AMC_IN (skipping check)"
  else
    PER_SHARD=$(( CALL_EST / (NUM_SHARDS > 0 ? NUM_SHARDS : 1) ))
    if (( PER_SHARD < MIN_PER_SHARD )); then
      echo "WARNING: ~$CALL_EST calls / $NUM_SHARDS shards = ~$PER_SHARD calls/shard (< $MIN_PER_SHARD)." >&2
      echo "         Shards this small make per-shard model reloads costly. Consider a smaller" >&2
      echo "         NUM_SHARDS (>= your max fleet size). Set SKIP_SHARD_CHECK=1 to silence." >&2
    else
      echo "Shard size: ~$PER_SHARD calls/shard (estimate from immediate subdirs of AMC_IN)"
    fi
  fi
fi

# Durable run config consumed by ops/resume_shard.sh -> ops/run_shard_no_docker.sh.
# NOTE: no INSTANCE_IDS here on purpose. Shard assignment is claim-based (lease pool on
# shared NFS), so instances can be recreated with new IDs and still resume. NUM_SHARDS is
# the fixed data-partition count for this run.
RUN_ENV_CONTENT="$(cat <<EOF
# AMC auto-resume run config (generated by ops/install_autoresume.sh)
export AWS_REGION='$AWS_REGION'
export REPO_DIR='$REPO_DIR'
export PYTHON_BIN='$PYTHON_BIN'
export HASH_MODE='$HASH_MODE'
export AMC_IN='$AMC_IN'
export RUN_ROOT='$RUN_ROOT'
export NUM_SHARDS='$NUM_SHARDS'
export STAGES='$STAGES'
export AUTO_PULL='$AUTO_PULL'
EOF
)"
RUN_ENV_B64="$(printf '%s\n' "$RUN_ENV_CONTENT" | base64 | tr -d '\n')"

# ---- Step A: single-writer setup on FIRST_ID (repo pull + write config) ----
PARAMS_A="$(python3 - "$REPO_DIR" "$RUN_ROOT" "$AMC_RUN_ENV" "$RUN_ENV_B64" "$AUTO_PULL" <<'PY'
import json, sys
repo_dir, run_root, run_env, b64, auto_pull = sys.argv[1:]
cmds = [
    "set -e",
    "export HOME=/root",
    f"mkdir -p '{run_root}' \"$(dirname '{run_env}')\"",
]
if auto_pull == "1":
    cmds += [
        f"git config --global --add safe.directory '{repo_dir}'",
        f"cd '{repo_dir}'",
        "git fetch origin",
        "git reset --hard origin/main",
    ]
cmds += [
    f"echo {b64} | base64 -d > '{run_root}/run.env'",
    f"cp '{run_root}/run.env' '{run_env}'",
    f"echo WROTE {run_env}:",
    f"cat '{run_env}'",
]
print(json.dumps({"commands": cmds}))
PY
)"

echo "== Step A: writing run config on $FIRST_ID =="
CMD_A="$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$FIRST_ID" \
  --document-name AWS-RunShellScript --timeout-seconds 180 \
  --parameters "$PARAMS_A" --query 'Command.CommandId' --output text)"
echo "  command-id: $CMD_A"

# Wait for Step A to finish before installing the service (which reads the config).
for _ in $(seq 1 60); do
  STATUS_A="$(aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$CMD_A" --instance-id "$FIRST_ID" --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "$STATUS_A" in
    Success) break ;;
    Failed|Cancelled|TimedOut) echo "Step A $STATUS_A; aborting" >&2;
      aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_A" --instance-id "$FIRST_ID" --query 'StandardErrorContent' --output text >&2 || true
      exit 3 ;;
  esac
  sleep 3
done
echo "  Step A status: ${STATUS_A:-unknown}"
aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_A" \
  --instance-id "$FIRST_ID" --query 'StandardOutputContent' --output text || true

# ---- Step B: install wrapper + unit and enable service on ALL online instances ----
PARAMS_B="$(python3 - "$REPO_DIR" <<'PY'
import json, sys
repo_dir = sys.argv[1]
cmds = [
    "set -e",
    f"install -m 0755 '{repo_dir}/ops/resume_shard.sh' /usr/local/bin/amc_resume_shard.sh",
    f"install -m 0644 '{repo_dir}/ops/amc-shard.service' /etc/systemd/system/amc-shard.service",
    "systemctl daemon-reload",
    "systemctl enable --now amc-shard.service",
    "systemctl --no-pager --full status amc-shard.service | head -n 15 || true",
]
print(json.dumps({"commands": cmds}))
PY
)"

echo "== Step B: install + enable service on all online instances =="
CMD_B="$(aws ssm send-command --region "$AWS_REGION" \
  --targets "Key=tag:Project,Values=$PROJECT_TAG" \
  --document-name AWS-RunShellScript \
  --max-concurrency "$NUM_SHARDS" --max-errors "$NUM_SHARDS" \
  --parameters "$PARAMS_B" --query 'Command.CommandId' --output text)"

echo
echo "Auto-resume installed."
echo "  Step A command-id: $CMD_A"
echo "  Step B command-id: $CMD_B"
echo
echo "Verify Step B across the fleet:"
echo "  aws ssm list-command-invocations --region $AWS_REGION --command-id $CMD_B --query 'CommandInvocations[].[InstanceId,Status]' --output text"
echo
echo "Watch shard progress (status JSON on shared NFS):"
echo "  aws ssm send-command --region $AWS_REGION --instance-ids $FIRST_ID --document-name AWS-RunShellScript \\"
echo "    --parameters '{\"commands\":[\"for f in $RUN_ROOT/status/shard-*.json; do cat \$f; echo; done\"]}' --query Command.CommandId --output text"
echo
echo "Stop the fleet:  touch $RUN_ROOT/STOP   (then optionally: systemctl disable --now amc-shard.service on each box)"
