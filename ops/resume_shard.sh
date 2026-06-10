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
# Config: a single env file (default /mnt/amc-data/amc-runs/active.env, on SHARED storage)
#   written by ops/install_autoresume.sh. It MUST be on the shared FS so every instance --
#   including spot replacements with new IDs -- can read it. Point elsewhere with
#   AMC_RUN_ENV. (A box-local default like /mnt/amc-runs/active.env is a trap: only the
#   single writer instance would have it; all other boxes time out waiting and exit.)
set -uo pipefail

RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-data/amc-runs/active.env}"
MAX_BACKOFF="${AMC_RESUME_MAX_BACKOFF:-300}"
CONFIG_WAIT_TRIES="${AMC_RESUME_CONFIG_WAIT_TRIES:-120}"   # x5s = up to 10 min for NFS/config
LEASE_TTL="${AMC_LEASE_TTL:-300}"                          # steal a shard if its lease is older than this
LEASE_HEARTBEAT="${AMC_LEASE_HEARTBEAT:-60}"               # refresh own lease this often
IDLE_RESCAN="${AMC_IDLE_RESCAN:-60}"                       # when all shards are owned by others, re-scan this often

log() { echo "[$(date -Is)] resume_shard: $*"; }
now_epoch() { date +%s; }

# ---- require the shared data mount ---------------------------------------------------
# This box can do NO work unless the shared FSx Lustre filesystem is mounted at
# $DATA_MOUNT: without it there is no repo, no run config, and no shard state. A box can
# boot "Running / healthy" yet have FAILED to mount (e.g. a kernel<->lustre-client kmod
# mismatch after an unattended kernel upgrade), in which case it would otherwise sit idle
# forever while the config-wait loop below silently spins. Fail fast and NON-ZERO so the
# unit's Restart=on-failure churns it and the box shows up as a FAILED service (visible to
# ops + diagnostics) instead of a silent idle "Running" instance.
#
# We allow a short grace window because FSx can attach a little after boot; a genuinely
# broken box (no kmod) never mounts and keeps failing/restarting, which is the goal.
DATA_MOUNT="${AMC_DATA_MOUNT:-/mnt/amc-data}"
MOUNT_WAIT_TRIES="${AMC_MOUNT_WAIT_TRIES:-24}"   # x5s = up to 120s for FSx to attach at boot
mtries=0
while ! timeout 15 mountpoint -q "$DATA_MOUNT" 2>/dev/null; do
  mtries=$((mtries + 1))
  if (( mtries > MOUNT_WAIT_TRIES )); then
    log "shared data mount $DATA_MOUNT NOT mounted after $((MOUNT_WAIT_TRIES * 5))s (FSx Lustre missing -- kernel/lustre-client kmod mismatch?); exiting non-zero so systemd restarts and this box stays visibly FAILED instead of idle"
    exit 1
  fi
  log "waiting for shared data mount $DATA_MOUNT to appear ($mtries/$MOUNT_WAIT_TRIES)"
  sleep 5
done

# ---- wait for shared storage + run config -------------------------------------------
# If the config is not visible yet (NFS still attaching, or the run not installed yet) we
# exit NON-ZERO so the systemd unit's Restart=on-failure relaunches us and we try again.
# Exiting 0 here would be read by systemd as a clean completion and the box would stay
# DEAD forever -- the exact failure mode that stranded spot-replacement boxes that were
# briefly pointed at the wrong path. Self-heal > silent give-up.
tries=0
while [[ ! -f "$RUN_ENV" ]]; do
  tries=$((tries + 1))
  if (( tries > CONFIG_WAIT_TRIES )); then
    log "no run config at $RUN_ENV after $((CONFIG_WAIT_TRIES * 5))s; will retry (systemd restart)"
    exit 1
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

# ---- ensure the per-stage Python environments exist (idempotent, build-once) --------
# setup_env.sh builds the main + align venvs on shared storage under a flock (exactly one
# builder; the rest wait for the ready marker) and then verifies imports + CUDA on THIS
# instance. On an already-provisioned fleet it is a fast no-op. We gate work on it so a
# freshly-recreated instance never claims a shard into a missing or broken environment.
ensure_env() {
  [[ "${AMC_SKIP_ENV_SETUP:-0}" == "1" ]] && return 0
  bash "$REPO_DIR/ops/setup_env.sh"
}
env_backoff=30
until ensure_env; do
  if stop_requested; then log "STOP sentinel present during env setup; exiting"; exit 0; fi
  log "environment not ready/verified; retrying in ${env_backoff}s"
  sleep "$env_backoff"
  env_backoff=$((env_backoff * 2)); (( env_backoff > MAX_BACKOFF )) && env_backoff="$MAX_BACKOFF"
done

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
