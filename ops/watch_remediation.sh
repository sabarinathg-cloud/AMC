#!/usr/bin/env bash
# Progress of the spoken-number remediation wave, read from shared FSx via one box.
#
# The driver logs one line per phase per shard, so the state of all 32 shards is
# readable from a single instance -- no need to poll each box.
#
#   bash ops/watch_remediation.sh            # one-shot
#   bash ops/watch_remediation.sh --loop     # refresh every 60s
set -uo pipefail

RUN_ROOT="${RUN_ROOT:-/mnt/amc-data/amc-runs/2025-full}"
LOOP=0
[[ "${1:-}" == "--loop" ]] && LOOP=1

pick_box() {
  aws ssm describe-instance-information \
    --filters "Key=tag:Project,Values=amc-ec2-fleet" \
    --query 'InstanceInformationList[?PingStatus==`Online`].InstanceId' \
    --output text 2>/tmp/awserr.txt | tr '\t' '\n' | head -1
}

remote() {  # $1 = instance id, $2 = script
  local b64 cmd status
  b64="$(printf '%s' "$2" | base64 | tr -d '\n')"
  cmd="$(aws ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript \
        --timeout-seconds 300 \
        --parameters "{\"commands\":[\"echo $b64 | base64 -d > /tmp/_wr.sh\",\"bash /tmp/_wr.sh\"]}" \
        --query 'Command.CommandId' --output text 2>/tmp/awserr.txt)"
  if [[ -z "$cmd" ]]; then
    echo "AWS error: $(tail -2 /tmp/awserr.txt)"
    return 1
  fi
  for _ in $(seq 1 60); do
    sleep 3
    status="$(aws ssm get-command-invocation --command-id "$cmd" --instance-id "$1" \
             --query 'Status' --output text 2>/dev/null)"
    [[ "$status" == Success || "$status" == Failed ]] && break
  done
  aws ssm get-command-invocation --command-id "$cmd" --instance-id "$1" \
    --query 'StandardOutputContent' --output text 2>/dev/null
}

SCRIPT='
R="'"$RUN_ROOT"'"
done_n=0; run_n=0; fail_n=0; todo_n=0
printf "%-9s %-11s %s\n" SHARD STATE DETAIL
for s in $(seq 0 31); do
  log="$R/logs/remediate-driver-$s.log"
  note="$R/outputs/shard-$s/.pii_pipeline/reports/spoken_number_remediation.json"
  slog="$R/logs/remediate-shard-$s.log"
  stage=""; last_fail=""
  if [ -f "$slog" ]; then
    stage=$(grep -oE "START [a-z_]+ shard" "$slog" 2>/dev/null | tail -1 | awk "{print \$2}")
    last=$(grep -oE "(START [a-z_]+ shard|END [a-z_]+ rc=[0-9]+)" "$slog" 2>/dev/null | tail -1)
    case "$last" in END*rc=0) ;; END*) last_fail="$last" ;; esac
  fi
  if [ -f "$log" ] && grep -q "DONE: every spoken number" "$log" 2>/dev/null; then
    done_n=$((done_n+1))
    detail=$(grep -oE "[0-9,]+ contain a spoken number, [0-9,]+ now masked, [0-9,]+ still unmasked" "$log" | tail -1)
    printf "%-9s %-11s %s\n" "shard-$s" VERIFIED "$detail"
  elif [ -f "$log" ] && [ -n "$last_fail" ]; then
    # A stage failure shows up as rc!=0 in the STAGE log; the driver dies on it under
    # `set -e`, so its own log just stops and would otherwise read as "running". Only
    # the LAST boundary counts: the log is appended across retries, so an old rc=2 sits
    # there forever once a retry is under way.
    fail_n=$((fail_n+1))
    printf "%-9s %-11s %s\n" "shard-$s" FAILED "$last_fail"
  elif [ -f "$log" ]; then
    run_n=$((run_n+1))
    prog=$(tail -c 400 "$slog" 2>/dev/null | tr "\r" "\n" | grep -oE "[0-9]+%\|" | tail -1 | tr -d "|")
    printf "%-9s %-11s %s\n" "shard-$s" running "stage=${stage:-?} ${prog:-}"
  elif [ -f "$note" ]; then
    # Remediated without the driver (the shard-0 pilot was run stage by stage).
    done_n=$((done_n+1))
    printf "%-9s %-11s %s\n" "shard-$s" VERIFIED "remediated directly, no driver log"
  else
    todo_n=$((todo_n+1))
    printf "%-9s %-11s %s\n" "shard-$s" "not started" ""
  fi
done
echo
echo "verified $done_n   running $run_n   failed $fail_n   not started $todo_n   of 32"
'

while :; do
  BOX="$(pick_box)"
  if [[ -z "$BOX" ]]; then
    echo "no online instances -- $(tail -2 /tmp/awserr.txt)"
  else
    clear 2>/dev/null || true
    echo "REMEDIATION  $RUN_ROOT   (via $BOX)   $(date -u +%H:%M:%SZ)"
    echo
    remote "$BOX" "$SCRIPT"
  fi
  (( LOOP )) || break
  sleep 60
done
