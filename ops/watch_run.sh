#!/usr/bin/env bash
# Fleet progress for an AMC pipeline run.
#
# Every worker writes its state to the SHARED run dir on FSx (status/shard-N.json,
# shards/shard-N.lock/lease, and the per-stage .done markers), so the whole fleet can be
# read from any single box. That makes this one SSM round trip regardless of how many
# machines are running -- polling each box individually would be 32x the calls and would
# hang on any box whose Lustre mount is wedged.
#
#   ops/watch_run.sh                    # one snapshot
#   ops/watch_run.sh -w                 # refresh until interrupted
#   ops/watch_run.sh -w -i 30           # refresh every 30s (default 60)
#   ops/watch_run.sh -r /mnt/amc-data/amc-runs/2026-full   # a different run
#
# Reads the active run config when -r is not given, so it follows whatever is running.
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-data/amc-runs/active.env}"
RUN_ROOT=""
INTERVAL=60
WATCH=0

while getopts "r:i:wh" opt; do
  case "$opt" in
    r) RUN_ROOT="$OPTARG" ;;
    i) INTERVAL="$OPTARG" ;;
    w) WATCH=1 ;;
    h) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "usage: $0 [-r RUN_ROOT] [-i SECONDS] [-w]" >&2; exit 2 ;;
  esac
done

# The remote payload. Kept as a heredoc (not a repo path) so this script works even when
# the shared checkout is mid-pull.
read -r -d '' REMOTE <<'REMOTE_EOF'
RUN_ENV_PATH="__RUN_ENV__"
RUN_ROOT_ARG="__RUN_ROOT__"
python3 - "$RUN_ENV_PATH" "$RUN_ROOT_ARG" <<'PY'
import glob, json, os, re, subprocess, sys, time

run_env, run_root_arg = sys.argv[1], sys.argv[2]


def from_env(path, key, default=""):
    try:
        text = open(path).read()
    except OSError:
        return default
    m = re.search(rf"^export {key}='([^']*)'", text, re.M)
    return m.group(1) if m else default


root = run_root_arg or from_env(run_env, "RUN_ROOT")
if not root:
    print("no run configured: pass -r RUN_ROOT")
    sys.exit(1)
num_shards = int(from_env(run_env, "NUM_SHARDS", "0") or 0)
stages = (from_env(run_env, "STAGES") or "").split()
if run_root_arg:
    # A run other than the active one keeps its own copy of the config.
    num_shards = int(from_env(f"{root}/run.env", "NUM_SHARDS", str(num_shards)) or num_shards)
    stages = (from_env(f"{root}/run.env", "STAGES") or " ".join(stages)).split()

now = time.time()
status = []
for f in glob.glob(f"{root}/status/shard-*.json"):
    try:
        status.append(json.load(open(f)))
    except Exception:
        pass

complete = [s for s in status if s.get("stage") == "complete"]
failed = [s for s in status if s.get("state") == "failed"]
active = [s for s in status if s.get("stage") != "complete" and s.get("state") != "failed"]

# Per-stage completion across every shard: the .done markers are the durable record.
marker_counts = {}
for m in glob.glob(f"{root}/outputs/shard-*/.pii_pipeline/stage_markers/*.done"):
    marker_counts[os.path.basename(m)[:-5]] = marker_counts.get(os.path.basename(m)[:-5], 0) + 1

# Progress in stage-units (shard x stage) so it is meaningful before any shard finishes.
total_units = num_shards * len(stages) if num_shards and stages else 0
done_units = sum(marker_counts.get(s, 0) for s in stages)

# Run start is run.env, written once at launch. Status files are NOT a substitute: they
# are rewritten at every stage transition, so their mtime is always recent and the
# throughput estimate would come out several times too optimistic.
started = None
for candidate in (f"{root}/run.env", f"{root}/status", root):
    try:
        started = os.path.getmtime(candidate)
        break
    except OSError:
        continue
elapsed_h = max((now - (started or now)) / 3600, 1e-6)

sizes = [int(s["input_files"]) for s in status if str(s.get("input_files", "")).isdigit()]
mean_calls = sum(sizes) / len(sizes) if sizes else 0
est_total_calls = mean_calls * num_shards if num_shards else 0
calls_done = sum(
    int(s["input_files"]) for s in complete if str(s.get("input_files", "")).isdigit()
)

print(f"RUN  {root}")
print(f"     {num_shards} shards x {len(stages)} stages   running {elapsed_h:.1f} h")
print()

pct = (done_units / total_units * 100) if total_units else 0
bar = "#" * int(pct / 2.5) + "." * (40 - int(pct / 2.5))
print(f"PROGRESS  [{bar}] {pct:5.1f}%   {done_units:,} / {total_units:,} stage-units")
if done_units and total_units:
    remaining_h = (total_units - done_units) / (done_units / elapsed_h)
    eta = time.strftime("%a %H:%M UTC", time.gmtime(now + remaining_h * 3600))
    print(f"          {done_units / elapsed_h:.0f} stage-units/h  ->  ~{remaining_h:.1f} h left  (ETA {eta})")
    if not complete:
        print("          (rough: stages are weighted equally, but they are not equally "
              "expensive.\n           Firms up once the first shard clears all stages.)")
print()

not_started = num_shards - len(status)
print(f"SHARDS    complete {len(complete):>4}   in flight {len(active):>3}   "
      f"not started {not_started:>4}   failed {len(failed):>3}")
if est_total_calls:
    print(f"CALLS     {calls_done:,} of ~{est_total_calls:,.0f} fully through all stages")
print()

if stages:
    print("STAGE COMPLETIONS (shards past each stage)")
    for s in stages:
        c = marker_counts.get(s, 0)
        width = int(c / num_shards * 30) if num_shards else 0
        print(f"  {s:<14} {c:>4}/{num_shards}  {'#' * width}")
    print()

if failed:
    print("FAILED SHARDS")
    for s in sorted(failed, key=lambda x: x.get("shard_index", 0)):
        print(f"  shard-{s.get('shard_index')} {s.get('hostname')} "
              f"{s.get('stage')}: {s.get('message', '')[:90]}")
    print()

# Lease age exposes a machine that died: its heartbeat stops and another box takes the
# shard over once the lease goes stale (default 300s).
leases = {}
for f in glob.glob(f"{root}/shards/shard-*.lock/lease"):
    idx = re.search(r"shard-(\d+)\.lock", f)
    try:
        parts = open(f).read().split()
        if idx and parts:
            leases[int(idx.group(1))] = now - float(parts[0])
    except Exception:
        pass

print(f"MACHINES ({len(active)} working)")
print(f"  {'host':<16}{'shard':>7}  {'calls':>6}  {'stage':<14}{'state':<10}{'in stage':>9}{'lease':>8}")
for s in sorted(active, key=lambda x: str(x.get("hostname"))):
    idx = s.get("shard_index")
    try:
        upd = time.mktime(time.strptime(s["updated_at"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        in_stage = f"{(now - upd) / 60:.0f}m"
    except Exception:
        in_stage = "?"
    age = leases.get(idx)
    lease = "-" if age is None else (f"{age:.0f}s" + ("!" if age > 300 else ""))
    print(f"  {str(s.get('hostname')):<16}{idx:>7}  {str(s.get('input_files','?')):>6}  "
          f"{str(s.get('stage')):<14}{str(s.get('state')):<10}{in_stage:>9}{lease:>8}")

print()
df = subprocess.run(["df", "-h", "/mnt/amc-data"], capture_output=True, text=True).stdout
if df:
    print("FSX  " + df.strip().splitlines()[-1])
PY
REMOTE_EOF

snapshot() {
  local ids id cmd status out
  ids="$(aws ssm describe-instance-information --region "$AWS_REGION" \
    --filters "Key=tag:Project,Values=$PROJECT_TAG" \
    --query 'InstanceInformationList[?PingStatus==`Online`].InstanceId' \
    --output text 2>/dev/null | tr '\t\n' '  ')"
  if [[ -z "${ids// }" ]]; then echo "no online instances tagged Project=$PROJECT_TAG" >&2; return 1; fi

  local payload
  payload="${REMOTE//__RUN_ENV__/$RUN_ENV}"
  payload="${payload//__RUN_ROOT__/$RUN_ROOT}"
  local b64; b64="$(printf '%s' "$payload" | base64 | tr -d '\n')"

  # Try a few boxes: a host whose Lustre mount is wedged never returns, and reading the
  # shared status dir from any healthy box gives the identical answer.
  local tried=0
  for id in $ids; do
    (( tried++ > 2 )) && break
    cmd="$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$id" \
      --document-name AWS-RunShellScript --timeout-seconds 120 \
      --parameters "{\"commands\":[\"echo $b64 | base64 -d | bash\"]}" \
      --query 'Command.CommandId' --output text 2>/dev/null)" || continue
    for _ in $(seq 1 30); do
      sleep 2
      status="$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$cmd" \
        --instance-id "$id" --query 'Status' --output text 2>/dev/null || echo Pending)"
      [[ "$status" == "Success" || "$status" == "Failed" ]] && break
    done
    if [[ "$status" == "Success" ]]; then
      aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$cmd" \
        --instance-id "$id" --query 'StandardOutputContent' --output text
      return 0
    fi
  done
  echo "could not read run status from any of the first $tried instances" >&2
  return 1
}

if (( WATCH )); then
  while true; do
    out="$(snapshot)"
    clear
    printf '%s\n' "$out"
    printf '\n(refreshing every %ss -- ctrl-c to stop)\n' "$INTERVAL"
    sleep "$INTERVAL"
  done
else
  snapshot
fi
