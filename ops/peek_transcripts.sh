#!/usr/bin/env bash
# Sample ASR output from a live shard to eyeball transcript quality mid-run.
#
# Reads a shard's state DB directly (the durable record every stage writes to), so this
# works while the shard is still being processed and costs nothing but one SSM call.
#
#   ops/peek_transcripts.sh                      # stats + samples, first ready shard
#   ops/peek_transcripts.sh -s 14                # a specific shard
#   ops/peek_transcripts.sh -m asr_qwen          # a different model (default asr_parakeet)
#   ops/peek_transcripts.sh -n 15                # more samples
#   ops/peek_transcripts.sh --raw                # do NOT mask digits (real PII on screen)
#
# Digits are replaced with # by default: these transcripts are pre-redaction, and card and
# account numbers are exactly what the run exists to remove.
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
RUN_ENV="${AMC_RUN_ENV:-/mnt/amc-data/amc-runs/active.env}"
SHARD=""
MODEL="parakeet"
SAMPLES=8
MASK=1
LANG_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) SHARD="$2"; shift 2 ;;
    -m) MODEL="${2#asr_}"; shift 2 ;;
    -n) SAMPLES="$2"; shift 2 ;;
    -l) LANG_FILTER="$2"; shift 2 ;;
    --raw) MASK=0; shift ;;
    -h) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "usage: $0 [-s SHARD] [-m MODEL] [-n SAMPLES] [-l LANG] [--raw]" >&2; exit 2 ;;
  esac
done

read -r -d '' REMOTE <<'REMOTE_EOF'
python3 - "__RUN_ENV__" "__SHARD__" "__MODEL__" "__SAMPLES__" "__MASK__" "__LANG__" <<'PY'
import glob, json, os, random, re, shutil, sqlite3, statistics, sys, tempfile

run_env, shard_arg, model, n_samples, mask, lang_filter = sys.argv[1:7]
n_samples, mask = int(n_samples), mask == "1"
lang_filter = lang_filter.lower()

m = re.search(r"^export RUN_ROOT='([^']*)'", open(run_env).read(), re.M)
root = m.group(1)

dbs = sorted(glob.glob(f"{root}/outputs/shard-*/.pii_pipeline/state/pipeline.sqlite3"))
if shard_arg:
    dbs = [d for d in dbs if f"/shard-{shard_arg}/" in d]
if not dbs:
    print("no shard databases found")
    sys.exit(1)
# Shards that finished the stage first: their DB is quiescent for this model's rows.
dbs.sort(key=lambda d: not os.path.exists(
    f"{os.path.dirname(os.path.dirname(d))}/stage_markers/asr_{model}.done"))

scratch = tempfile.mkdtemp(prefix="amc-peek-")


def rows_for(db):
    """Query a LOCAL copy of the shard DB.

    The owning box is writing this file right now, and SQLite's WAL coordinates through
    shared memory that does not work across nodes -- reading the Lustre copy in place
    risks a torn read or a lock error. Copying db+wal and letting SQLite recover the WAL
    locally gives a clean point-in-time snapshot instead.
    """
    local = os.path.join(scratch, os.path.basename(db))
    for suffix in ("", "-wal"):
        try:
            shutil.copyfile(db + suffix, local + suffix)
        except OSError:
            pass
    con = sqlite3.connect(local)
    con.row_factory = sqlite3.Row
    try:
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT model_name FROM model_results").fetchall()]
        match = next((n for n in names if model in n), None)
        if match is None:
            return names, []
        return names, con.execute(
            "SELECT r.segment_id, r.model_name, r.payload_json AS rp, s.payload_json AS sp "
            "FROM model_results r JOIN segments s ON s.segment_id = r.segment_id "
            "WHERE r.model_name = ?",
            (match,),
        ).fetchall()
    finally:
        con.close()


db, rows, seen_names = None, [], []
for candidate in dbs:
    try:
        names, found = rows_for(candidate)
    except sqlite3.Error as exc:
        print(f"  ({os.path.basename(os.path.dirname(os.path.dirname(candidate)))}: {exc})")
        continue
    seen_names = names or seen_names
    if found:
        db, rows = candidate, found
        break
if not rows:
    print(f"no '{model}' results yet in {len(dbs)} shard db(s); models present: {seen_names}")
    sys.exit(0)

shard = db.split("/outputs/")[1].split("/")[0]
print(f"SHARD {shard}   model={rows[0]['model_name']}   {len(rows):,} segments transcribed")
print()

empties = errors = 0
lengths, rates, langs = [], [], {}
empty_durs = []
records = []
for r in rows:
    rp = json.loads(r["rp"])
    sp = json.loads(r["sp"])
    text = (rp.get("transcript") or "").strip()
    dur = float(sp.get("duration_sec") or 0)
    if rp.get("error"):
        errors += 1
        continue
    if not text:
        empties += 1
        empty_durs.append(dur)
        continue
    if lang_filter and lang_filter != (rp.get("language") or "").lower():
        continue
    lengths.append(len(text))
    if dur > 0:
        rates.append(len(text) / dur)
    lang = (rp.get("language") or "?").lower()
    langs[lang] = langs.get(lang, 0) + 1
    records.append((r["segment_id"], sp.get("call_id"), dur, text))

total = len(rows)
print(f"  empty transcripts  {empties:>6}  ({empties / total * 100:.1f}%)", end="")
if empty_durs:
    # Empties on sub-second clips are VAD picking up a breath; empties on long clips
    # would mean real speech was dropped, which is the case worth worrying about.
    long_empty = sum(1 for d in empty_durs if d > 2.0)
    print(f"   median {statistics.median(empty_durs):.1f}s, {long_empty} over 2s")
else:
    print()
print(f"  hard errors        {errors:>6}  ({errors / total * 100:.1f}%)")
if lengths:
    print(f"  chars/segment      median {statistics.median(lengths):.0f}   p90 {sorted(lengths)[int(len(lengths) * 0.9)]:.0f}   max {max(lengths)}")
if rates:
    srt = sorted(rates)
    print(f"  chars/sec          median {statistics.median(rates):.1f}   p90 {srt[int(len(srt) * 0.9)]:.1f}   max {max(rates):.1f}   (>25 suggests looping)")
print(f"  language           {', '.join(f'{k}={v}' for k, v in sorted(langs.items(), key=lambda x: -x[1]))}")

# Same shape of check the consensus guard applies, so the count here previews how much
# this model would lose to the degeneracy filter.
def repeat_ratio(text):
    w = text.lower().split()
    if len(w) < 12:
        return 0.0
    grams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(grams)) / len(grams)


suspect = [rec for rec in records if rec[2] > 0 and (len(rec[3]) / rec[2] > 25 or repeat_ratio(rec[3]) > 0.75)]
print(f"  looks degenerate   {len(suspect):>6}  ({len(suspect) / max(len(records), 1) * 100:.1f}%)  <- dropped before consensus")
print()


def show(text, limit=320):
    if mask:
        text = re.sub(r"\d", "#", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


random.seed(0)
print(f"--- {min(n_samples, len(records))} RANDOM SEGMENTS " + "-" * 40)
for seg, call, dur, text in random.sample(records, min(n_samples, len(records))):
    print(f"\n[{dur:5.1f}s] {call}")
    print(f"  {show(text)}")

longest = sorted(records, key=lambda r: -(len(r[3]) / r[2] if r[2] else 0))[:3]
print(f"\n--- 3 FASTEST (most chars/sec -- where looping shows up) " + "-" * 12)
for seg, call, dur, text in longest:
    print(f"\n[{dur:5.1f}s, {len(text) / dur if dur else 0:.0f} c/s] {call}")
    print(f"  {show(text, 240)}")

shutil.rmtree(scratch, ignore_errors=True)
PY
REMOTE_EOF

ids="$(aws ssm describe-instance-information --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'InstanceInformationList[?PingStatus==`Online`].InstanceId' \
  --output text 2>/dev/null | tr '\t\n' '  ')"

payload="${REMOTE//__RUN_ENV__/$RUN_ENV}"
payload="${payload//__SHARD__/$SHARD}"
payload="${payload//__MODEL__/$MODEL}"
payload="${payload//__SAMPLES__/$SAMPLES}"
payload="${payload//__MASK__/$MASK}"
payload="${payload//__LANG__/$LANG_FILTER}"
b64="$(printf '%s' "$payload" | base64 | tr -d '\n')"

tried=0
for id in $ids; do
  (( tried++ > 2 )) && break
  cmd="$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$id" \
    --document-name AWS-RunShellScript --timeout-seconds 180 \
    --parameters "{\"commands\":[\"echo $b64 | base64 -d | bash\"]}" \
    --query 'Command.CommandId' --output text 2>/dev/null)" || continue
  for _ in $(seq 1 45); do
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
echo "could not read transcripts from any of the first $tried instances" >&2
exit 1
