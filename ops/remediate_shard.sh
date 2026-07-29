#!/usr/bin/env bash
# Re-run pii -> manifest for ONE finished shard so its spoken numbers get masked.
#
# 32 shards cleared `pii` before the spoken-number detector existed. This drives the
# real runner over the affected items only: remediate_spoken_numbers.py deletes the
# artifacts whose spans changed, and every stage then skips the ~99% of items it still
# has an artifact for. Measured on shard-0: pii 54s, align 5m, mask_plan 9s.
#
#   RUN_ROOT=/mnt/amc-data/amc-runs/2025-full bash ops/remediate_shard.sh 7
#
# CLAIMS the shard through the same lease protocol resume_shard.sh uses, and holds it
# until done. That is not optional: clearing the stage markers makes a finished shard
# look incomplete, so an idle worker will claim and re-run it. On the shard-0 pilot a
# worker did exactly that 30 minutes in, and two pipelines wrote one sqlite file.
set -Eeuo pipefail
umask 0002

SHARD="${1:?usage: remediate_shard.sh SHARD_INDEX}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
LEASE_TTL="${AMC_LEASE_TTL:-300}"
PY="${AMC_MAIN_PY:-/mnt/amc-data/venvs/main/bin/python}"
[[ -x "$PY" ]] || PY=python3

SHARD_DIR="$RUN_ROOT/outputs/shard-$SHARD"
NOTE="$SHARD_DIR/.pii_pipeline/reports/spoken_number_remediation.json"
LOG="$RUN_ROOT/logs/remediate-shard-$SHARD.log"
mkdir -p "$RUN_ROOT/logs"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remediate[$SHARD]: $*"; }

[[ -d "$SHARD_DIR" ]] || { say "no such shard dir: $SHARD_DIR"; exit 2; }

if [[ -f "$NOTE" ]]; then
  say "SKIP: already remediated ($NOTE)"
  exit 0
fi

# --- claim the shard, same protocol as resume_shard.sh -----------------------
SHARDS_DIR="$RUN_ROOT/shards"
LOCK="$SHARDS_DIR/shard-$SHARD.lock"
SELF="$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || hostname):$$"
mkdir -p "$SHARDS_DIR"

write_lease() { printf '%s %s %s\n' "$(date +%s)" "$SELF" "$(hostname)" > "$LOCK/lease"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  EPOCH="$(awk 'NR==1{print $1}' "$LOCK/lease" 2>/dev/null || true)"
  [[ -n "$EPOCH" ]] || EPOCH="$(stat -c %Y "$LOCK" 2>/dev/null || echo 0)"
  AGE=$(( $(date +%s) - EPOCH ))
  if (( AGE >= 0 && AGE < LEASE_TTL )); then
    say "SKIP: worker $(awk 'NR==1{print $3}' "$LOCK/lease" 2>/dev/null) holds a live lease (${AGE}s old)"
    exit 3
  fi
  say "taking over a stale lease (${AGE}s old)"
  TOMB="$LOCK.dead.$$.$RANDOM"
  mv "$LOCK" "$TOMB" 2>/dev/null && rm -rf "$TOMB"
  mkdir "$LOCK" 2>/dev/null || { say "SKIP: lost the takeover race"; exit 3; }
fi
write_lease
HB_PID=""
( while true; do write_lease; sleep "${AMC_LEASE_HEARTBEAT:-60}"; done ) & HB_PID=$!
cleanup() {
  [[ -n "$HB_PID" ]] && kill "$HB_PID" 2>/dev/null || true
  rm -rf "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
say "claimed shard (lease held by $SELF)"

cd "$REPO_DIR"
say "invalidating changed items"
"$PY" ops/remediate_spoken_numbers.py --run "$RUN_ROOT" --shard "$SHARD" --apply

say "re-running pii -> manifest (log: $LOG)"
set -a
# shellcheck disable=SC1091
. "$RUN_ROOT/run.env"
set +a
export STAGES="pii align mask_plan redact validate manifest"
export SHARD_INDEX="$SHARD"
# The inputs are already resident for a finished shard; skip the restore pass.
export AMC_PREWARM="${AMC_PREWARM:-0}"
bash ops/run_shard_no_docker.sh >>"$LOG" 2>&1

say "verifying"
if "$PY" ops/remediate_spoken_numbers.py --run "$RUN_ROOT" --shard "$SHARD" --verify; then
  say "DONE: every spoken number in this shard is masked"
else
  say "WARNING: some spoken numbers are still unmasked -- see above"
  exit 1
fi
