# AMC PII Audio Pipeline

This is my pipeline for AMC audio preprocessing, ASR, normalization, PII
detection, force alignment, channel-specific audio masking, validation, and ASR
training manifests.

The main goal is simple:

- keep the original folder structure
- process each channel separately
- detect transcript-based PII
- align the detected PII back to audio time
- mask only the affected channel and time range
- keep unmasked mono channel segments for ASR training
- keep enough state so I can resume after failures

If my input is:

```text
/mnt/amc-data/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

then my redacted output will be:

```text
<output>/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

The pipeline state, cache, and reports will be inside:

```text
<output>/.pii_pipeline/
```

I use Python 3.10.12 for this project.

## Install

```bash
cd /mnt/amc-data-ebs/AMC
git pull

pyenv local 3.10.12
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -e .
python3.10 -m pip install -e ".[qwen]"
```

For Qwen ASR I keep the same supported runtime used in my training notebook:

```text
qwen-asr==0.0.6
transformers==4.57.6
```

I check the machine like this:

```bash
python3.10 --version
nvidia-smi
ffmpeg -version
```

## My Folder Rule

I always pass an input root that contains the year folder.

Correct:

```text
/mnt/amc-data/2023/<call_id>/audio.opus
--input /mnt/amc-data
```

For one-call or one-year testing, I create a small staging folder that still has
the same `year/call_id/audio.opus` layout.

I do not point `--input` directly at `/mnt/amc-data/2023`, because then the
pipeline cannot see the year level correctly.

## One Call Test

This is the command set I use to test one call end to end.

```bash
export YEAR=2023
export CALL_ID=aaf5285a-3d70-56a8-87e3-77c26d547494
export SRC_ROOT=/mnt/amc-data
export ONE_IN=/tmp/amc-one
export ONE_OUT=/tmp/amc-one-output

rm -rf "$ONE_IN" "$ONE_OUT"
mkdir -p "$ONE_IN/$YEAR/$CALL_ID"
cp "$SRC_ROOT/$YEAR/$CALL_ID/audio.opus" "$ONE_IN/$YEAR/$CALL_ID/audio.opus"
```

First I check that discovery works:

```bash
python3.10 -m amc_pipeline.cli dry-run \
  --input "$ONE_IN" \
  --output "$ONE_OUT"
```

Then I run the stages one by one. On one A10 GPU, I normally run one ASR model
per command so memory is cleared before the next model starts.

```bash
python3.10 -m amc_pipeline.cli run-stage preprocess \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --vad-backend silero

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models whisper

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models qwen \
  --asr-batch-sizes qwen=1

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models cohere \
  --asr-batch-sizes cohere=1

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models granite \
  --asr-batch-sizes granite=1

python3.10 -m amc_pipeline.cli run-stage normalize \
  --input "$ONE_IN" \
  --output "$ONE_OUT"

python3.10 -m amc_pipeline.cli run-stage consensus \
  --input "$ONE_IN" \
  --output "$ONE_OUT"

python3.10 -m amc_pipeline.cli run-stage pii \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --detectors regex,gliner,piiranha,spacy,rule_name,saved_json

python3.10 -m amc_pipeline.cli run-stage align \
  --input "$ONE_IN" \
  --output "$ONE_OUT"

python3.10 -m amc_pipeline.cli run-stage mask-plan \
  --input "$ONE_IN" \
  --output "$ONE_OUT"

python3.10 -m amc_pipeline.cli run-stage redact \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --mask-strategy beep \
  --allow-fallback-format wav

python3.10 -m amc_pipeline.cli run-stage validate \
  --input "$ONE_IN" \
  --output "$ONE_OUT"

python3.10 -m amc_pipeline.cli run-stage manifest \
  --input "$ONE_IN" \
  --output "$ONE_OUT"
```

To listen in Jupyter:

```python
from IPython.display import Audio, display

display(Audio("/tmp/amc-one-output/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus"))
```

## One Folder Or One Year

If my folder already contains year folders, I use it directly:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-output

python3.10 -m amc_pipeline.cli dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"
```

If I want to process only one year, I make a staging root with a symlink:

```bash
export YEAR=2023
export SRC_ROOT=/mnt/amc-data
export YEAR_IN=/tmp/amc-year-$YEAR
export YEAR_OUT=/mnt/amc-output-$YEAR

rm -rf "$YEAR_IN"
mkdir -p "$YEAR_IN"
ln -s "$SRC_ROOT/$YEAR" "$YEAR_IN/$YEAR"
```

Then I use `"$YEAR_IN"` as `--input` and `"$YEAR_OUT"` as `--output`.

## Full Folder On One Machine

If I want to run the full dataset on one machine in one command:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted

python3.10 -m amc_pipeline.cli run \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models whisper,qwen,cohere,granite \
  --asr-batch-sizes qwen=1,cohere=1,granite=1 \
  --detectors regex,gliner,piiranha,spacy,rule_name,saved_json \
  --vad-backend silero \
  --mask-strategy beep \
  --allow-fallback-format wav
```

For production I prefer stage by stage, because it is easier to verify and
resume:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted

python3.10 -m amc_pipeline.cli run-stage preprocess --input "$AMC_IN" --output "$AMC_OUT" --vad-backend silero
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models whisper
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models qwen --asr-batch-sizes qwen=1
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models cohere --asr-batch-sizes cohere=1
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models granite --asr-batch-sizes granite=1
python3.10 -m amc_pipeline.cli run-stage normalize --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli run-stage consensus --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli run-stage pii --input "$AMC_IN" --output "$AMC_OUT" --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
python3.10 -m amc_pipeline.cli run-stage align --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli run-stage mask-plan --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli run-stage redact --input "$AMC_IN" --output "$AMC_OUT" --mask-strategy beep --allow-fallback-format wav
python3.10 -m amc_pipeline.cli run-stage validate --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli run-stage manifest --input "$AMC_IN" --output "$AMC_OUT"
```

## Resume After Failure

If anything fails, I rerun the failed stage with the same `--input` and
`--output`. The state is already stored inside `<output>/.pii_pipeline/`.

Examples:

```bash
python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models qwen \
  --asr-batch-sizes qwen=1

python3.10 -m amc_pipeline.cli run-stage align \
  --input "$AMC_IN" \
  --output "$AMC_OUT"

python3.10 -m amc_pipeline.cli run-stage redact \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --mask-strategy beep \
  --allow-fallback-format wav
```

If I need to pause a long preprocessing run:

```bash
python3.10 -m amc_pipeline.cli pause --output "$AMC_OUT" --global
```

## Outputs I Check

```text
<output>/<year>/<call_id>/audio.opus
<output>/<year>/segments/<call_id>/*.wav
<output>/<year>/manifests/segments.csv
<output>/<year>/manifests/segments.jsonl
<output>/manifests/all_segments.csv
<output>/manifests/all_segments.jsonl
<output>/manifests/all_segments.parquet
<output>/manifests/review.xlsx
<output>/.pii_pipeline/state/pipeline.sqlite3  # SQLite mode only
<output>/.pii_pipeline/reports/
```

## Multiple A10 GPU Machines With Shared Storage

I use one of these two patterns. I do not run the exact same stage and same
model from multiple machines into the same SQLite output folder.

That unsafe pattern can duplicate work and cause database locking issues on
shared storage.

### Option A: One Shared Run, One ASR Model Per Machine

I use this when I have up to four A10 machines and want all machines to work on
the same dataset.

The rule is:

- preprocess once
- run one ASR model per machine
- wait until all ASR jobs finish
- run the remaining stages once

For this shared run I use Postgres state, not SQLite.

Install the Postgres client package on every machine:

```bash
python3.10 -m pip install "psycopg[binary]>=3.1"
```

Create this config on shared storage as `config.shared.yaml`:

```yaml
input_root: /mnt/amc-data
output_root: /mnt/amc-redacted
run_id: amc-prod
progress_enabled: true
languages: [en, es]

audio:
  vad_backend: silero
  silero_repo_or_dir: snakers4/silero-vad
  target_sample_rate: 16000
  target_segment_sec: 12
  max_segment_sec: 30
  min_segment_sec: 0.25
  merge_gap_sec: 0.8
  min_split_gap_sec: 0.35

masking:
  strategy: beep
  pre_padding_ms: 80
  post_padding_ms: 120
  allow_wav_fallback: true

alignment:
  backend: whisperx

state:
  backend: postgres
  postgres_dsn: postgresql://amc_user:amc_password@POSTGRES_HOST:5432/amc_pipeline
```

On every machine I set:

```bash
cd /mnt/amc-data-ebs/AMC
git pull

export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted
export CFG=/mnt/amc-data-ebs/AMC/config.shared.yaml
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

First I run discovery and preprocessing once from one machine:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"

python3.10 -m amc_pipeline.cli --config "$CFG" run-stage preprocess \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --vad-backend silero
```

Then I run these in parallel, one command per A10 machine.

Machine 1:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models whisper
```

Machine 2:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models qwen \
  --asr-batch-sizes qwen=1
```

Machine 3:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models cohere \
  --asr-batch-sizes cohere=1
```

Machine 4:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models granite \
  --asr-batch-sizes granite=1
```

After all four ASR jobs finish, I run the rest once from one machine:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage normalize --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage consensus --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage pii --input "$AMC_IN" --output "$AMC_OUT" --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage align --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage mask-plan --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage redact --input "$AMC_IN" --output "$AMC_OUT" --mask-strategy beep --allow-fallback-format wav
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage validate --input "$AMC_IN" --output "$AMC_OUT"
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage manifest --input "$AMC_IN" --output "$AMC_OUT"
```

Important rules for this mode:

- I run preprocessing once.
- I run each ASR model on exactly one machine.
- I do not run `qwen` on two machines against the same output.
- I run post-ASR stages only once after all ASR jobs finish.
- I use Postgres state for this shared multi-machine mode.

### Option B: N-Way Sharded Runs

I use this when I have many A10 machines or when I want the safest parallel
setup. Each machine gets its own subset of call IDs, its own output folder, and
its own state database.

This avoids shared database writes during processing.

Create call-ID shards on shared storage:

```bash
export AMC_IN=/mnt/amc-data
export SHARD_ROOT=/mnt/amc-shards
export N=4

rm -rf "$SHARD_ROOT"
for i in $(seq 0 $((N - 1))); do
  mkdir -p "$SHARD_ROOT/shard-$i"
done

python3.10 - <<'PY'
import hashlib
import os
from pathlib import Path

src = Path(os.environ["AMC_IN"])
dst = Path(os.environ["SHARD_ROOT"])
n = int(os.environ["N"])

for audio in sorted(src.glob("*/*/audio.*")):
    rel = audio.relative_to(src)
    if len(rel.parts) < 3:
        continue
    year, call_id = rel.parts[0], rel.parts[1]
    shard = int(hashlib.sha1(call_id.encode("utf-8")).hexdigest(), 16) % n
    target_dir = dst / f"shard-{shard}" / year / call_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / audio.name
    if not target.exists():
        target.symlink_to(audio)
PY
```

On each machine I use a different `SHARD_ID`.

Machine example:

```bash
export SHARD_ID=0
export SHARD_IN=/mnt/amc-shards/shard-$SHARD_ID
export SHARD_OUT=/mnt/amc-output-shards/shard-$SHARD_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3.10 -m amc_pipeline.cli run \
  --input "$SHARD_IN" \
  --output "$SHARD_OUT" \
  --models whisper,qwen,cohere,granite \
  --asr-batch-sizes qwen=1,cohere=1,granite=1 \
  --detectors regex,gliner,piiranha,spacy,rule_name,saved_json \
  --vad-backend silero \
  --mask-strategy beep \
  --allow-fallback-format wav
```

Then I repeat with `SHARD_ID=1`, `SHARD_ID=2`, etc.

To collect only the final redacted call audio into one folder:

```bash
export FINAL_OUT=/mnt/amc-redacted-final
mkdir -p "$FINAL_OUT"

for d in /mnt/amc-output-shards/shard-*; do
  rsync -a \
    --exclude '/.pii_pipeline/' \
    --include '*/' \
    --include 'audio.*' \
    --exclude '*' \
    "$d"/ "$FINAL_OUT"/
done
```

I keep the shard output folders if I need the full state, reports, training
segments, and manifests from each shard.

## Commands I Use To Inspect Output

Final redacted files:

```bash
find "$AMC_OUT" -path '*/audio.*' -type f | head -20
```

Segment manifests:

```bash
ls -lh "$AMC_OUT/manifests"
head -5 "$AMC_OUT/manifests/all_segments.csv"
```

Stage reports:

```bash
find "$AMC_OUT/.pii_pipeline/reports" -maxdepth 1 -type f -print
```

SQLite failures:

```bash
python3.10 - <<'PY'
import os
import sqlite3
from pathlib import Path

db = Path(os.environ["AMC_OUT"]) / ".pii_pipeline/state/pipeline.sqlite3"
conn = sqlite3.connect(db)
for row in conn.execute("select scope, scope_id, error from failures order by created_at desc limit 20"):
    print(row)
PY
```

Progress bars are enabled by default. If I want quieter logs, I add
`--no-progress` to the command.
