#!/usr/bin/env bash
set -Eeuo pipefail

# Pre-warm (HSM-restore) an FSx-for-Lustre tree from its linked S3 bucket so that
# every file's DATA is resident on Lustre before processing.
#
# Why: FSx releases cold data to S3 to save space (lfs hsm_state shows "released").
# The first read of a released file triggers a synchronous restore that holds the
# file's layout lock; readers park in the kernel at `ldlm_completion_ast` until the
# S3 fetch lands. At fleet scale this serialises discovery and ASR behind FSx's
# bounded restore concurrency and looks like a hang. Critically, the discovery
# builder reads EVERY metadata.json's content, so even a metadata-only walk wedges
# on a released tree. Restoring up front turns the whole run into pure local-Lustre
# reads -- no per-read stalls, full 32-box use.
#
# Distributed mode (AMC_PREWARM_NUM_SHARDS>1): each box restores only its 1/N slice
# of the top-level directories (ls + NR%N==i). The union of all slices is the whole
# tree, so a fleet barrier (in run_shard_no_docker.sh) that waits for all N slices
# guarantees the whole tree is resident before the single discovery builder walks it.
# This splits the otherwise ~1h single-box metadata walk N ways and uses every box.
#
# Restores are filesystem-wide: any client that restores a file makes it resident
# for ALL clients, so slice ownership is purely a work-partition, not a correctness
# constraint -- a file restored by one box is resident for whichever box processes it.
#
# Usage:
#   ops/prewarm_hsm.sh <dir>                                  # whole tree (single box)
#   AMC_PREWARM_NUM_SHARDS=32 AMC_PREWARM_SHARD_INDEX=7 \
#     ops/prewarm_hsm.sh /mnt/amc-data/2022                   # this box's 1/32 slice
#
# Exit codes: 0 = slice fully resident (or nothing to do / not an HSM fs);
#             1 = timed out with files still released.

ROOT="${1:?usage: prewarm_hsm.sh <input_dir>}"
PARALLEL="${AMC_PREWARM_PARALLEL:-24}"      # concurrent lfs invocations
BATCH="${AMC_PREWARM_BATCH:-200}"           # files per lfs invocation
POLL_SEC="${AMC_PREWARM_POLL:-15}"          # residency re-check interval
TIMEOUT_SEC="${AMC_PREWARM_TIMEOUT:-21600}" # 6h hard cap
STATUS_FILE="${AMC_PREWARM_STATUS:-}"       # optional: write progress here for ops
NUM_SHARDS="${AMC_PREWARM_NUM_SHARDS:-1}"   # >1 => only enumerate/restore this slice
SHARD_INDEX="${AMC_PREWARM_SHARD_INDEX:-0}"

log() { echo "[$(date -Is)] prewarm[$SHARD_INDEX/$NUM_SHARDS]: $*"; }

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
  printf '{"state":"%s","released_remaining":%s,"total_files":%s,"shard_index":%s,"num_shards":%s,"updated_at":"%s"}\n' \
    "$1" "${2:-0}" "${3:-0}" "$SHARD_INDEX" "$NUM_SHARDS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$STATUS_FILE" 2>/dev/null || true
}

# Print the paths (one per line) of files in $1 that are currently HSM-released.
# Split on ": (0x" so paths containing spaces survive.
released_subset() {  # $1 = file list
  xargs -r -a "$1" -P "$PARALLEL" -n "$BATCH" lfs hsm_state 2>/dev/null \
    | awk -F': \\(0x' '/released/{print $1}'
}

# --- enumerate files in scope ------------------------------------------------
write_status "scanning" 0 0
if [[ "$NUM_SHARDS" -gt 1 ]]; then
  # Partition the top-level directories across the fleet. ls is a single readdir of
  # the parent (cheap relative to walking the whole tree); awk keeps every Nth entry
  # so index 0..N-1 jointly cover all directories exactly once.
  log "listing top-level dirs under $ROOT and taking slice $SHARD_INDEX/$NUM_SHARDS..."
  ls -1 "$ROOT" 2>/dev/null | awk -v n="$NUM_SHARDS" -v i="$SHARD_INDEX" 'NR % n == i' \
    > "$WORK/dirs.txt" || true
  ndirs="$(wc -l < "$WORK/dirs.txt" | tr -d ' ')"
  log "slice owns $ndirs top-level dirs; enumerating their files (parallel=$PARALLEL)..."
  # One find per dir, parallelised. Each call dir holds only a few files, so thousands
  # of tiny finds in parallel are far cheaper than one serial walk of the whole tree.
  xargs -r -a "$WORK/dirs.txt" -P "$PARALLEL" -I {} find "$ROOT/{}" -type f 2>/dev/null \
    > "$ALL" || true
else
  log "scanning whole tree $ROOT for files..."
  find "$ROOT" -type f > "$ALL" 2>/dev/null || true
fi

total="$(wc -l < "$ALL" | tr -d ' ')"
log "files in scope: $total"
if [[ "$total" -eq 0 ]]; then
  write_status "done" 0 0
  exit 0
fi

# --- restore -----------------------------------------------------------------
# Restore EVERY file in scope up front (hsm_restore is a no-op on already-resident
# files, so we skip a separate residency pre-scan and start fetching immediately),
# then poll the still-released subset until it drains.
log "issuing HSM restore for $total files (parallel=$PARALLEL batch=$BATCH)..."
write_status "restoring" "$total" "$total"
xargs -r -a "$ALL" -P "$PARALLEL" -n "$BATCH" lfs hsm_restore 2>/dev/null || true

cp "$ALL" "$REL"
deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while :; do
  # Re-check ONLY the still-released set so each pass shrinks and stays cheap even
  # on a large slice.
  released_subset "$REL" > "$REL.next" || true
  mv "$REL.next" "$REL"
  rem="$(wc -l < "$REL" | tr -d ' ')"
  done_n=$(( total - rem ))
  log "resident $done_n/$total ; released remaining=$rem"
  write_status "restoring" "$rem" "$total"
  if [[ "$rem" -eq 0 ]]; then
    log "PREWARM COMPLETE: all $total files in slice resident"
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
