#!/usr/bin/env bash
set -Eeuo pipefail

# Pre-warm (HSM-restore) an FSx-for-Lustre tree from its linked S3 bucket so that
# every file's DATA is resident on Lustre before processing.
#
# Why: FSx releases cold data to S3 to save space (lfs hsm_state shows "released").
# The first read of a released file triggers a synchronous restore that holds the
# file's layout lock; readers park in the kernel at `ldlm_completion_ast` until the
# S3 fetch lands. At fleet scale this serialises discovery and ASR behind FSx's
# bounded restore concurrency and looks like a hang. Restoring up front turns the
# whole run into pure local-Lustre reads -- no per-read stalls, full 32-box use.
#
# Restores are filesystem-wide: any client that restores a file makes it resident
# for ALL clients, so this is safe to run from a single box and correctness does
# not depend on which box later processes which shard.
#
# Usage:
#   ops/prewarm_hsm.sh <dir>
#   AMC_PREWARM_PARALLEL=24 AMC_PREWARM_BATCH=200 ops/prewarm_hsm.sh /mnt/amc-data/2022
#
# Exit codes: 0 = tree fully resident (or nothing to do / not an HSM fs);
#             1 = timed out with files still released.

ROOT="${1:?usage: prewarm_hsm.sh <input_dir>}"
PARALLEL="${AMC_PREWARM_PARALLEL:-24}"     # concurrent lfs invocations
BATCH="${AMC_PREWARM_BATCH:-200}"          # files per lfs invocation
POLL_SEC="${AMC_PREWARM_POLL:-15}"         # residency re-check interval
TIMEOUT_SEC="${AMC_PREWARM_TIMEOUT:-21600}" # 6h hard cap
STATUS_FILE="${AMC_PREWARM_STATUS:-}"      # optional: write progress here for ops

log() { echo "[$(date -Is)] prewarm: $*"; }

# Not a Lustre client / no HSM tooling -> nothing to pre-warm; succeed quietly so
# this is safe to call unconditionally on any filesystem (tests, NFS, local).
if ! command -v lfs >/dev/null 2>&1; then
  log "lfs not found; assuming non-HSM filesystem -- nothing to pre-warm for $ROOT"
  exit 0
fi
if [[ ! -d "$ROOT" ]]; then
  log "input dir $ROOT does not exist; nothing to pre-warm"
  exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/amc-prewarm.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
ALL="$WORK/all.txt"
REL="$WORK/released.txt"

write_status() {  # $1=state $2=remaining $3=total
  [[ -n "$STATUS_FILE" ]] || return 0
  mkdir -p "$(dirname "$STATUS_FILE")" 2>/dev/null || true
  printf '{"state":"%s","released_remaining":%s,"total_files":%s,"updated_at":"%s"}\n' \
    "$1" "${2:-0}" "${3:-0}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE" 2>/dev/null || true
}

# Print the paths (one per line) of files in $1 that are currently HSM-released.
# Split on ": (0x" so paths containing spaces survive.
released_subset() {  # $1 = file list
  xargs -r -a "$1" -P "$PARALLEL" -n "$BATCH" lfs hsm_state 2>/dev/null \
    | awk -F': \\(0x' '/released/{print $1}'
}

log "scanning $ROOT for files..."
find "$ROOT" -type f > "$ALL" 2>/dev/null || true
total="$(wc -l < "$ALL" | tr -d ' ')"
log "total files: $total"
if [[ "$total" -eq 0 ]]; then
  write_status "done" 0 0
  exit 0
fi

log "checking residency (this scans all files once)..."
released_subset "$ALL" > "$REL" || true
rem="$(wc -l < "$REL" | tr -d ' ')"
log "released (need restore): $rem of $total"
write_status "restoring" "$rem" "$total"
if [[ "$rem" -eq 0 ]]; then
  log "all files already resident; nothing to do"
  write_status "done" 0 "$total"
  exit 0
fi

log "issuing HSM restore for $rem files (parallel=$PARALLEL batch=$BATCH)..."
xargs -r -a "$REL" -P "$PARALLEL" -n "$BATCH" lfs hsm_restore 2>/dev/null || true

deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while :; do
  # Re-check ONLY the still-released set so each pass shrinks and stays cheap even
  # on very large trees.
  released_subset "$REL" > "$REL.next" || true
  mv "$REL.next" "$REL"
  rem="$(wc -l < "$REL" | tr -d ' ')"
  done_n=$(( total - rem ))
  log "resident $done_n/$total ; released remaining=$rem"
  write_status "restoring" "$rem" "$total"
  if [[ "$rem" -eq 0 ]]; then
    log "PREWARM COMPLETE: all $total files resident"
    write_status "done" 0 "$total"
    exit 0
  fi
  if (( $(date +%s) >= deadline )); then
    log "PREWARM TIMEOUT after ${TIMEOUT_SEC}s with $rem files still released"
    write_status "timeout" "$rem" "$total"
    exit 1
  fi
  # Re-issue restore on the laggards (idempotent) in case any request was dropped.
  xargs -r -a "$REL" -P "$PARALLEL" -n "$BATCH" lfs hsm_restore 2>/dev/null || true
  sleep "$POLL_SEC"
done
