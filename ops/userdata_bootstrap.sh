#!/usr/bin/env bash
# EC2 user-data / cloud-init bootstrap for AMC auto-resume.
#
# Put this in the launch template / launch configuration "user data" so that EVERY newly
# created instance self-installs the auto-resume service on first boot. This is required
# when instances are recreated (new instance IDs, fresh disks) on a schedule: the local
# systemd unit and wrapper script do not survive recreation, but the repo and run config
# live on shared NFS and do.
#
# It is intentionally self-contained and idempotent: safe to run on every boot.
#
# Assumptions:
#   - The shared NFS is (or will be) mounted and contains REPO_DIR below.
#   - ops/install_autoresume.sh has already written /mnt/amc-runs/active.env once
#     (the durable run config). Until that exists, the service simply idles and rechecks.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
AMC_RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-runs/active.env}"

log() { echo "[$(date -Is)] amc-bootstrap: $*"; }

# Wait for the shared mount holding the repo (NFS may attach a little after boot).
for _ in $(seq 1 120); do
  [[ -f "$REPO_DIR/ops/resume_shard.sh" && -f "$REPO_DIR/ops/amc-shard.service" ]] && break
  sleep 5
done
if [[ ! -f "$REPO_DIR/ops/resume_shard.sh" ]]; then
  log "repo not available at $REPO_DIR after wait; cannot install auto-resume"
  exit 0
fi

install -m 0755 "$REPO_DIR/ops/resume_shard.sh" /usr/local/bin/amc_resume_shard.sh
install -m 0644 "$REPO_DIR/ops/amc-shard.service" /etc/systemd/system/amc-shard.service

# Pass a non-default run-config location through to the service, if set.
if [[ "$AMC_RUN_ENV" != "/mnt/amc-runs/active.env" ]]; then
  mkdir -p /etc/systemd/system/amc-shard.service.d
  cat > /etc/systemd/system/amc-shard.service.d/override.conf <<EOF
[Service]
Environment=AMC_RUN_ENV=$AMC_RUN_ENV
EOF
fi

systemctl daemon-reload
systemctl enable --now amc-shard.service
log "auto-resume service installed and started"
