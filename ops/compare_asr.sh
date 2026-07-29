#!/usr/bin/env bash
# Put two models' transcripts of the same segment side by side.
#
# Reads a live shard's state DB, so this works mid-run and needs nothing but one SSM call.
#
#   ops/compare_asr.sh                       # agreement stats + a random sample
#   ops/compare_asr.sh -k worst              # the biggest disagreements
#   ops/compare_asr.sh -k empty              # where one model said nothing
#   ops/compare_asr.sh -k loop               # where one model degenerated
#   ops/compare_asr.sh -a parakeet -b granite -n 12
#   ops/compare_asr.sh -s 14 -w 140          # a shard, and a wider terminal
#   ops/compare_asr.sh --raw                 # do NOT mask digits (real PII on screen)
#
# Digits are replaced with # by default: these transcripts are pre-redaction.
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-data/amc-runs/active.env}"
SHARD=""
LEFT="parakeet"
RIGHT="qwen"
KIND="random"
SAMPLES=8
WIDTH="${COLUMNS:-120}"
MASK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) SHARD="$2"; shift 2 ;;
    -a) LEFT="${2#asr_}"; shift 2 ;;
    -b) RIGHT="${2#asr_}"; shift 2 ;;
    -k) KIND="$2"; shift 2 ;;
    -n) SAMPLES="$2"; shift 2 ;;
    -w) WIDTH="$2"; shift 2 ;;
    --raw) MASK=0; shift ;;
    -h) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "usage: $0 [-s SHARD] [-a MODEL] [-b MODEL] [-k random|worst|empty|loop] [-n N] [-w COLS] [--raw]" >&2; exit 2 ;;
  esac
done

read -r -d '' REMOTE <<'REMOTE_EOF'
python3 - "__RUN_ENV__" "__SHARD__" "__LEFT__" "__RIGHT__" "__KIND__" "__SAMPLES__" "__WIDTH__" "__MASK__" <<'PY'
import difflib, glob, json, os, random, re, shutil, sqlite3, sys, tempfile, textwrap

run_env, shard_arg, left, right, kind, n_samples, width, mask = sys.argv[1:9]
n_samples, width, mask = int(n_samples), int(width), mask == "1"

m = re.search(r"^export RUN_ROOT='([^']*)'", open(run_env).read(), re.M)
root = m.group(1)

dbs = sorted(glob.glob(f"{root}/outputs/shard-*/.pii_pipeline/state/pipeline.sqlite3"))
if shard_arg:
    dbs = [d for d in dbs if f"/shard-{shard_arg}/" in d]

scratch = tempfile.mkdtemp(prefix="amc-cmp-")


def load(db):
    """Read a LOCAL copy: the owning box is writing this file and WAL shared memory
    does not work across nodes, so reading it in place risks a torn read."""
    local = os.path.join(scratch, "p.sqlite3")
    for suffix in ("", "-wal"):
        try:
            shutil.copyfile(db + suffix, local + suffix)
        except OSError:
            pass
    con = sqlite3.connect(local)
    try:
        segs = {s: json.loads(p) for s, p in con.execute(
            "SELECT segment_id, payload_json FROM segments")}
        by = {}
        for s, name, p in con.execute(
                "SELECT segment_id, model_name, payload_json FROM model_results"):
            by.setdefault(s, {})[name] = (json.loads(p).get("transcript") or "").strip()
        return segs, by
    finally:
        con.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(local + suffix)
            except OSError:
                pass


db, segs, by = None, {}, {}
for candidate in dbs:
    try:
        s, b = load(candidate)
    except sqlite3.Error:
        continue
    if any(left in v and right in v for v in b.values()):
        db, segs, by = candidate, s, b
        break
if db is None:
    print(f"no shard has both '{left}' and '{right}' yet")
    sys.exit(0)

pairs = [(s, v[left], v[right]) for s, v in by.items() if left in v and right in v]
shard = db.split("/outputs/")[1].split("/")[0]
print(f"SHARD {shard}   {left} vs {right}   {len(pairs):,} segments transcribed by both")
print()


def norm(t):
    """Compare like the normalize stage does -- casing and punctuation are not
    disagreement, so counting them as such would drown out the real differences."""
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def sim(a, b):
    """Word-level ratio, autojunk OFF.

    Both matter. Comparing CHARACTERS with difflib's default autojunk silently destroys
    the score on long transcripts: past 200 elements it treats any element occurring in
    >1% of the sequence as junk, which for characters means the common letters, so two
    near-identical 30-second transcripts can score 0.02.
    """
    a, b = norm(a).split(), norm(b).split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def repeat_ratio(t):
    w = t.lower().split()
    if len(w) < 12:
        return 0.0
    g = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(g)) / len(g)


scored = [(s, a, b, sim(a, b)) for s, a, b in pairs]
identical = sum(1 for _, a, b, r in scored if norm(a) == norm(b))
close = sum(1 for _, a, b, r in scored if 0.8 <= r < 1.0 or (r == 1.0 and norm(a) != norm(b)))
partial = sum(1 for *_, r in scored if 0.5 <= r < 0.8)
different = sum(1 for *_, r in scored if r < 0.5)
one_empty = sum(1 for _, a, b, _ in scored if bool(norm(a)) != bool(norm(b)))
total = len(scored)


def pct(n):
    return f"({n / total * 100:5.1f}%)" if total else ""


print("AGREEMENT (case and punctuation ignored)")
print(f"  identical          {identical:>7}  {pct(identical)}")
print(f"  close  >=0.8       {close:>7}  {pct(close)}")
print(f"  partial 0.5-0.8    {partial:>7}  {pct(partial)}")
print(f"  different <0.5     {different:>7}  {pct(different)}")
print(f"  one side empty     {one_empty:>7}  {pct(one_empty)}")
print(f"  mean similarity    {sum(r for *_, r in scored) / total:>7.3f}" if total else "")
print()

# Whether disagreement matters depends entirely on WHERE it sits. Two models differing on
# a half-second "Mm" is noise; differing on a 20s stretch of account details is not.
print("SIMILARITY BY SEGMENT LENGTH")
buckets = [("under 1s", 0, 1), ("1-2s", 1, 2), ("2-5s", 2, 5), ("5-15s", 5, 15), ("over 15s", 15, 1e9)]
for label, lo, hi in buckets:
    group = [x for x in scored
             if lo <= float(segs.get(x[0], {}).get("duration_sec") or 0) < hi]
    if not group:
        continue
    mean = sum(r for *_, r in group) / len(group)
    bad = sum(1 for *_, r in group if r < 0.5)
    print(f"  {label:<10} {len(group):>7} segments   mean {mean:.3f}   "
          f"under 0.5: {bad / len(group) * 100:4.1f}%")
print()

if kind == "worst":
    # Rank by disagreement WEIGHTED BY DURATION. Sorting on similarity alone just surfaces
    # half-second backchannels ("Mm" vs "Uh huh") that score 0.00 and mean nothing; the
    # disagreements worth reading are the long segments where real content diverges.
    def weight(x):
        dur = float(segs.get(x[0], {}).get("duration_sec") or 0)
        return (1 - x[3]) * min(dur, 30)

    picked = sorted([x for x in scored if norm(x[1]) and norm(x[2])],
                    key=weight, reverse=True)[:n_samples]
    title = "BIGGEST DISAGREEMENTS (weighted by segment length)"
elif kind == "empty":
    picked = [x for x in scored if bool(norm(x[1])) != bool(norm(x[2]))][:n_samples]
    title = "ONE SIDE EMPTY"
elif kind == "loop":
    picked = [x for x in scored if repeat_ratio(x[1]) > 0.75 or repeat_ratio(x[2]) > 0.75][:n_samples]
    title = "DEGENERATE / LOOPING ON ONE SIDE"
else:
    random.seed(0)
    picked = random.sample(scored, min(n_samples, total))
    title = f"{min(n_samples, total)} RANDOM SEGMENTS"

col = max((width - 5) // 2, 30)
print(f"--- {title} " + "-" * max(width - len(title) - 5, 0))
print(f"  {left.upper():<{col}} | {right.upper()}")

for seg, a, b, r in picked:
    dur = float(segs.get(seg, {}).get("duration_sec") or 0)
    call = segs.get(seg, {}).get("call_id", "?")
    print()
    print(f"  [{dur:.1f}s] {call}   similarity {r:.2f}")
    if mask:
        a, b = re.sub(r"\d", "#", a), re.sub(r"\d", "#", b)
    # Cap runaway loops so one degenerate row cannot flood the screen.
    a, b = " ".join(a.split())[:600], " ".join(b.split())[:600]
    la = textwrap.wrap(a, col) or ["(empty)"]
    lb = textwrap.wrap(b, col) or ["(empty)"]
    for i in range(max(len(la), len(lb))):
        lhs = la[i] if i < len(la) else ""
        rhs = lb[i] if i < len(lb) else ""
        print(f"  {lhs:<{col}} | {rhs}")

shutil.rmtree(scratch, ignore_errors=True)
PY
REMOTE_EOF

ids="$(aws ssm describe-instance-information --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'InstanceInformationList[?PingStatus==`Online`].InstanceId' \
  --output text 2>/dev/null | tr '\t\n' '  ')"

payload="${REMOTE//__RUN_ENV__/$RUN_ENV}"
payload="${payload//__SHARD__/$SHARD}"
payload="${payload//__LEFT__/$LEFT}"
payload="${payload//__RIGHT__/$RIGHT}"
payload="${payload//__KIND__/$KIND}"
payload="${payload//__SAMPLES__/$SAMPLES}"
payload="${payload//__WIDTH__/$WIDTH}"
payload="${payload//__MASK__/$MASK}"
b64="$(printf '%s' "$payload" | base64 | tr -d '\n')"

tried=0
for id in $ids; do
  (( tried++ > 2 )) && break
  cmd="$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$id" \
    --document-name AWS-RunShellScript --timeout-seconds 300 \
    --parameters "{\"commands\":[\"echo $b64 | base64 -d | bash\"]}" \
    --query 'Command.CommandId' --output text 2>/dev/null)" || continue
  for _ in $(seq 1 75); do
    sleep 2
    status="$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$cmd" \
      --instance-id "$id" --query 'Status' --output text 2>/dev/null || echo Pending)"
    [[ "$status" == "Success" || "$status" == "Failed" ]] && break
  done
  if [[ "$status" == "Success" ]]; then
    aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$cmd" \
      --instance-id "$id" --query 'StandardOutputContent' --output text
    exit 0
  fi
  [[ "$status" == "Failed" ]] && aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$cmd" --instance-id "$id" --query 'StandardErrorContent' --output text
done
echo "could not read from any of the first $tried instances" >&2
exit 1
