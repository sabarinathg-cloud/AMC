#!/usr/bin/env bash
# Boot-time / crash resume worker for the AMC pipeline.
#
# Why this exists:
#   `ssm send-command` is one-shot. If an instance reboots, is spot-reclaimed, or is
#   OOM-killed mid-run, nothing relaunches the pipeline. This wrapper is started by a
#   systemd unit at boot (and restarted on failure) and drives shards to completion.
#
# Instance IDs are NOT stable here:
#   The fleet may be torn down and recreated (e.g. nightly), so instance IDs change.
#   Therefore shard assignment is decoupled from instance identity. The unit of work is
#   the *shard index* (a deterministic call-id hash partition whose state lives at
#   $RUN_ROOT/outputs/shard-N on shared NFS). Each running instance atomically CLAIMS an
#   incomplete shard from a shared lease pool, runs it, then claims the next. A new
#   instance with a new ID simply picks up whatever shard is free or whose previous owner
#   has gone silent (stale lease) and resumes that partition's state in place.
#
# Idempotent resume:
#   run_shard_no_docker.sh writes a per-stage marker only after that stage exits 0, and
#   every stage skips already-finished work via the per-shard SQLite state. So claiming a
#   shard and re-running it continues exactly where the previous owner left off.
#
# Config: a single env file (default /mnt/amc-runs/active.env) written by
#   ops/install_autoresume.sh. Point elsewhere with AMC_RUN_ENV.
set -uo pipefail

RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-runs/active.env}"
MAX_BACKOFF="${AMC_RESUME_MAX_BACKOFF:-300}"
CONFIG_WAIT_TRIES="${AMC_RESUME_CONFIG_WAIT_TRIES:-120}"   # x5s = up to 10 min for NFS/config
LEASE_TTL="${AMC_LEASE_TTL:-300}"                          # steal a shard if its lease is older than this
LEASE_HEARTBEAT="${AMC_LEASE_HEARTBEAT:-60}"               # refresh own lease this often
IDLE_RESCAN="${AMC_IDLE_RESCAN:-60}"                       # when all shards are owned by others, re-scan this often

log() { echo "[$(date -Is)] resume_shard: $*"; }
now_epoch() { date +%s; }

# ---- wait for shared storage + run config -------------------------------------------
tries=0
while [[ ! -f "$RUN_ENV" ]]; do
  tries=$((tries + 1))
  if (( tries > CONFIG_WAIT_TRIES )); then
    log "no run config at $RUN_ENV after $((CONFIG_WAIT_TRIES * 5))s; nothing to resume"
    exit 0
  fi
  sleep 5
done

set -a
# shellcheck disable=SC1090
source "$RUN_ENV"
set +a

: "${REPO_DIR:=/mnt/amc-data/AMC}"; export REPO_DIR
if [[ -z "${RUN_ROOT:-}" ]]; then log "RUN_ROOT missing in $RUN_ENV; abort"; exit 1; fi
if [[ -z "${NUM_SHARDS:-}" ]]; then log "NUM_SHARDS missing in $RUN_ENV; abort"; exit 1; fi
export RUN_ROOT NUM_SHARDS

INSTANCE_ID="${INSTANCE_ID:-$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || hostname)}"
export INSTANCE_ID
SELF="$INSTANCE_ID:$$"

SHARDS_DIR="$RUN_ROOT/shards"
mkdir -p "$SHARDS_DIR"

stop_requested() { [[ -f "$RUN_ROOT/STOP" ]]; }

# A shard is "done" when its final status row says stage=complete/state=completed.
shard_done() {
  local i="$1" sj="$RUN_ROOT/status/shard-$i.json"
  [[ -f "$sj" ]] || return 1
  python3 - "$sj" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get("stage") == "complete" and d.get("state") == "completed" else 1)
PY
}

lease_path() { echo "$SHARDS_DIR/shard-$1.lock/lease"; }

lease_epoch() {  # prints heartbeat epoch of a lease, or empty
  local f; f="$(lease_path "$1")"
  [[ -f "$f" ]] && awk 'NR==1{print $1}' "$f" 2>/dev/null
}
lease_owner() {
  local f; f="$(lease_path "$1")"
  [[ -f "$f" ]] && awk 'NR==1{print $2}' "$f" 2>/dev/null
}

write_lease() {  # $1=shard index
  local f; f="$(lease_path "$1")"
  printf '%s %s %s\n' "$(now_epoch)" "$SELF" "$(hostname)" > "$f"
}

# Try to claim shard $1. Echoes "ok" on success.
try_claim() {
  local i="$1" lock="$SHARDS_DIR/shard-$i.lock"
  if mkdir "$lock" 2>/dev/null; then          # atomic on NFS: fresh claim
    write_lease "$i"; echo ok; return 0
  fi
  # lock exists: resume our own, or steal if stale
  local owner ep age
  owner="$(lease_owner "$i")"
  if [[ "$owner" == "$SELF" ]]; then
    write_lease "$i"; echo ok; return 0
  fi
  ep="$(lease_epoch "$i")"
  if [[ -z "$ep" ]]; then ep=0; fi
  age=$(( $(now_epoch) - ep ))
  if (( age > LEASE_TTL )); then
    # Atomically take over a stale lock: only one concurrent rename wins.
    local tomb="$lock.dead.$$.$RANDOM"
    if mv "$lock" "$tomb" 2>/dev/null; then
      rm -rf "$tomb" 2>/dev/null || true
      if mkdir "$lock" 2>/dev/null; then write_lease "$i"; echo ok; return 0; fi
    fi
  fi
  return 1
}

release_lease() { rm -rf "$SHARDS_DIR/shard-$1.lock" 2>/dev/null || true; }

# Background heartbeat keeps our lease fresh while a shard runs.
HB_PID=""
start_heartbeat() {
  local i="$1"
  ( while true; do write_lease "$i"; sleep "$LEASE_HEARTBEAT"; done ) &
  HB_PID=$!
}
stop_heartbeat() {
  [[ -n "$HB_PID" ]] && kill "$HB_PID" 2>/dev/null || true
  HB_PID=""
}
cleanup() { stop_heartbeat; }
trap cleanup EXIT INT TERM

backoff=15
while true; do
  if stop_requested; then log "STOP sentinel present; exiting"; exit 0; fi

  # Are there any shards left to do?
  remaining=0
  claimed=""
  for ((i=0; i<NUM_SHARDS; i++)); do
    if shard_done "$i"; then continue; fi
    remaining=$((remaining + 1))
    if [[ -z "$claimed" ]] && [[ "$(try_claim "$i" || true)" == "ok" ]]; then
      claimed="$i"
    fi
  done

  if (( remaining == 0 )); then
    log "all $NUM_SHARDS shards complete; nothing left to do"
    exit 0
  fi

  if [[ -z "$claimed" ]]; then
    # Work remains but it is all owned by live instances. Idle and rescan so we can take
    # over if one of them dies (its lease will go stale).
    log "no claimable shard right now (remaining=$remaining owned by others); rescan in ${IDLE_RESCAN}s"
    sleep "$IDLE_RESCAN"
    continue
  fi

  log "claimed shard $claimed of $NUM_SHARDS (self=$SELF); starting"
  export SHARD_INDEX="$claimed"
  start_heartbeat "$claimed"
  bash "$REPO_DIR/ops/run_shard_no_docker.sh"
  rc=$?
  stop_heartbeat

  if [[ "$rc" -eq 0 ]]; then
    log "shard $claimed completed (rc=0)"
    release_lease "$claimed"      # done; status file marks it complete for everyone
    backoff=15
    continue                       # look for another incomplete shard
  fi

  log "shard $claimed exited rc=$rc; releasing lease so another instance can take over; retry in ${backoff}s"
  release_lease "$claimed"
  unset SHARD_INDEX
  sleep "$backoff"
  backoff=$((backoff * 2)); (( backoff > MAX_BACKOFF )) && backoff="$MAX_BACKOFF"
done
