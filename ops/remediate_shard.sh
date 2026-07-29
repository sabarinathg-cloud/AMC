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
# Refuses to touch a shard another box holds a live lease on -- two pipelines writing
# one shard's sqlite would corrupt it -- and refuses to redo an already-remediated one.
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

# A live lease means a worker is mid-shard on it right now.
LEASE="$RUN_ROOT/shards/shard-$SHARD.lock/lease"
if [[ -f "$LEASE" ]]; then
  EPOCH="$(grep -oE '[0-9]{9,}' "$LEASE" | head -1 || true)"
  if [[ -n "$EPOCH" ]]; then
    AGE=$(( $(date +%s) - EPOCH ))
    if (( AGE >= 0 && AGE < LEASE_TTL )); then
      say "SKIP: a worker holds a live lease (${AGE}s old); shard is in flight"
      exit 3
    fi
  fi
fi

if [[ -f "$NOTE" ]]; then
  say "SKIP: already remediated ($NOTE)"
  exit 0
fi

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
