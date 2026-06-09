#!/usr/bin/env bash
# Durable, serverless fleet auto-heal via SSM State Manager.
#
# Creates a State Manager association that AUTO-APPLIES to every instance tagged
# Project=$PROJECT_TAG -- including brand-new ASG/Spot replacements the moment they register
# with SSM -- and, if amc-shard.service is not active, installs+starts it from the shared repo
# (ops/userdata_bootstrap.sh). After this runs once, blank replacements self-join the lease
# pool with NO human action, ever. The schedule re-checks as a safety net.
#
# Requires: ssm:CreateAssociation (one-time grant to the operator role). Everything else
# (SendCommand to tagged instances, the shared repo + venvs on Lustre) is already in place.
#
# Run ONCE:
#   AWS_REGION=us-east-1 bash ops/autoheal_association.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
ASSOC_NAME="${ASSOC_NAME:-amc-autoresume-ensure}"
# Re-check cadence. Kept tight (5 min) so a freshly-registered ASG/Spot replacement
# self-installs the service within ~5 min instead of waiting up to a full schedule
# interval. The applied command is a guarded no-op on healthy boxes (it exits early if
# amc-shard.service is already active), so frequent runs are cheap.
SCHEDULE="${SCHEDULE:-rate(5 minutes)}"

echo "Creating SSM State Manager association '$ASSOC_NAME' on Project=$PROJECT_TAG ..."
aws ssm create-association --region "$AWS_REGION" \
  --name AWS-RunShellScript \
  --association-name "$ASSOC_NAME" \
  --targets "Key=tag:Project,Values=$PROJECT_TAG" \
  --parameters "{\"commands\":[\"systemctl is-active --quiet amc-shard.service && echo already-active && exit 0\",\"bash $REPO_DIR/ops/userdata_bootstrap.sh\"]}" \
  --schedule-expression "$SCHEDULE" \
  --max-concurrency "100%" --max-errors "100%" \
  --compliance-severity UNSPECIFIED \
  --query 'AssociationDescription.{Name:AssociationName,Id:AssociationId,Status:Overview.Status}' --output table

cat <<EOF

Created. New/blank instances tagged $PROJECT_TAG will now self-install the shard service
automatically. Verify any time with:

  aws ssm describe-association --region $AWS_REGION --association-name $ASSOC_NAME \\
    --query '{Status:Overview.Status,Targets:Targets}' --output json

Remove it (e.g. after the run completes) with:

  aws ssm delete-association --region $AWS_REGION --association-name $ASSOC_NAME
EOF
