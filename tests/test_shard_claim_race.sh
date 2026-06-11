#!/usr/bin/env bash
# Concurrency test for the shard-claim lease pool in ops/resume_shard.sh.
#
# Guards the race fixed in try_claim(): a claim is `mkdir lock` (atomic) THEN
# write_lease (non-atomic). If a second box reads the lock in that window and a
# missing lease is treated as epoch 0, age becomes ~now (>> LEASE_TTL) and the
# second box "steals" a brand-new claim -> multiple owners for one shard. This
# manifested live as 3 boxes on shard 0 / 2 on shard 6 after a simultaneous
# 32-box `systemctl restart`, leaving high-index shards unclaimed.
#
# The claim logic below is a faithful copy of ops/resume_shard.sh (keep in sync).
# CLAIM_DELAY widens the mkdir->write_lease window so the race is exercised
# deterministically rather than depending on real NFS timing.
set -uo pipefail

LEASE_TTL="${AMC_LEASE_TTL:-300}"
CLAIM_DELAY="${CLAIM_DELAY:-0.15}"

now_epoch() { date +%s; }
lease_path() { echo "$SHARDS_DIR/shard-$1.lock/lease"; }
lease_epoch() { local f; f="$(lease_path "$1")"; [[ -f "$f" ]] && awk 'NR==1{print $1}' "$f" 2>/dev/null; }
lease_owner() { local f; f="$(lease_path "$1")"; [[ -f "$f" ]] && awk 'NR==1{print $2}' "$f" 2>/dev/null; }
write_lease() { local f; f="$(lease_path "$1")"; printf '%s %s %s\n' "$(now_epoch)" "$SELF" "$(hostname)" > "$f"; }

# Fixed try_claim (mtime fallback when the lease file is not yet present).
try_claim() {
  local i="$1" lock="$SHARDS_DIR/shard-$i.lock"
  if mkdir "$lock" 2>/dev/null; then
    # Simulate the real, non-zero gap between claiming the lock and writing the
    # lease (slow NFS / scheduling). This is where the race lives.
    sleep "$CLAIM_DELAY"
    write_lease "$i"; echo ok; return 0
  fi
  local owner ep age
  owner="$(lease_owner "$i")"
  if [[ "$owner" == "$SELF" ]]; then write_lease "$i"; echo ok; return 0; fi
  ep="$(lease_epoch "$i")"
  if [[ -z "$ep" ]]; then
    ep="$(stat -c %Y "$lock" 2>/dev/null || stat -f %m "$lock" 2>/dev/null || echo 0)"
  fi
  age=$(( $(now_epoch) - ep ))
  if (( age > LEASE_TTL )); then
    local tomb="$lock.dead.$$.$RANDOM"
    if mv "$lock" "$tomb" 2>/dev/null; then
      rm -rf "$tomb" 2>/dev/null || true
      if mkdir "$lock" 2>/dev/null; then sleep "$CLAIM_DELAY"; write_lease "$i"; echo ok; return 0; fi
    fi
  fi
  return 1
}

# One worker: scan shards 0..NUM-1, claim the FIRST available (mirrors the real
# single-claim-per-pass loop), record the claim.
worker() {
  local self="$1" num="$2" results="$3"
  export SELF="$self"
  local i
  for ((i=0; i<num; i++)); do
    if [[ "$(try_claim "$i" || true)" == "ok" ]]; then
      echo "$i $self" >> "$results"
      return 0
    fi
  done
  return 0
}

run_trial() {
  local workers="$1" shards="$2"
  SHARDS_DIR="$(mktemp -d)"
  local results="$SHARDS_DIR/.results"
  : > "$results"
  local pids=()
  local w
  for ((w=0; w<workers; w++)); do
    worker "self-$w:$RANDOM" "$shards" "$results" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  # Every claimed shard must have exactly one owner; total distinct claims must
  # equal min(workers, shards).
  local dup distinct expected
  dup="$(awk '{print $1}' "$results" | sort | uniq -d)"
  distinct="$(awk '{print $1}' "$results" | sort -u | wc -l | tr -d ' ')"
  expected=$(( workers < shards ? workers : shards ))
  rm -rf "$SHARDS_DIR"

  if [[ -n "$dup" ]]; then
    echo "FAIL: shard(s) claimed by >1 worker: $(echo "$dup" | tr '\n' ' ')"
    return 1
  fi
  if [[ "$distinct" != "$expected" ]]; then
    echo "FAIL: expected $expected distinct claims, got $distinct"
    return 1
  fi
  return 0
}

main() {
  local trials="${TRIALS:-8}"
  local t
  for ((t=1; t<=trials; t++)); do
    # Heavy oversubscription (2x) maximizes contention on the low shards, the
    # exact pattern that produced the live triple-claim.
    if ! run_trial 16 8; then
      echo "trial $t/$trials FAILED"
      exit 1
    fi
    # Also the fleet-shaped case: workers == shards (the live 32==32 layout).
    if ! run_trial 12 12; then
      echo "trial $t/$trials (NxN) FAILED"
      exit 1
    fi
  done
  echo "OK: $((trials * 2)) trials, no double-claims, full distinct coverage"
}

main "$@"
