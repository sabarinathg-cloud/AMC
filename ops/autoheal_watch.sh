#!/usr/bin/env bash
# Interim fleet auto-heal watchdog.
#
# Why: ASG replacements for Spot-reclaimed boxes come up BLANK (the launch-template user-data
# does not install amc-shard.service), so they sit idle until someone installs the service.
# This watchdog removes the manual step using only ssm:SendCommand (which the operator already
# has) -- it does NOT need ssm:CreateAssociation or any EC2 launch-template permission.
#
# What it does: every INTERVAL seconds it tells every instance tagged Project=$PROJECT_TAG:
# "if amc-shard.service is not active, install+start it from the shared repo". The check is a
# no-op on healthy boxes (incl. the busy ones) and only heals blank/new replacements, which
# then verify venvs and claim a free/stale shard on their own.
#
# This is a STOPGAP. The durable, serverless fix is ops/autoheal_association.sh (needs
# ssm:CreateAssociation) or baking ops/userdata_bootstrap.sh into the launch template (DevOps).
#
# Run it on any always-on host you control (laptop/bastion/small box):
#   AWS_REGION=us-east-1 nohup bash ops/autoheal_watch.sh >/tmp/amc_autoheal.log 2>&1 &
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
INTERVAL="${INTERVAL:-120}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"

log() { echo "[$(date -Is)] autoheal: $*"; }

log "watching Project=$PROJECT_TAG every ${INTERVAL}s (region=$AWS_REGION, repo=$REPO_DIR)"
while true; do
  CID="$(aws ssm send-command --region "$AWS_REGION" \
    --targets "Key=tag:Project,Values=$PROJECT_TAG" \
    --document-name AWS-RunShellScript --max-concurrency 100% --max-errors 100% \
    --comment "amc autoheal sweep" \
    --parameters "commands=[\"systemctl is-active --quiet amc-shard.service && exit 0\",\"bash $REPO_DIR/ops/userdata_bootstrap.sh\",\"echo AUTOHEALED-\$(hostname -s)\"]" \
    --query 'Command.CommandId' --output text 2>/dev/null)" || {
      log "send-command failed (creds/region?); retrying in ${INTERVAL}s"
      sleep "$INTERVAL"; continue
    }
  log "sweep dispatched cmd=$CID"
  sleep "$INTERVAL"
done
