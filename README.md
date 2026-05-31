# AMC PII Audio Pipeline

Resumable audio preprocessing, ASR, normalization, transcript PII detection,
forced alignment, channel-specific masking, validation, and manifest generation.

The pipeline preserves the input folder structure in the output folder. For an
input like:

```text
/mnt/amc-data/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

the redacted output is written to:

```text
<output>/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

Pipeline state, cache, and reports are written under:

```text
<output>/.pii_pipeline/
```

Use Python 3.10.12.

## Install

```bash
cd /mnt/amc-data-ebs/AMC
git pull

pyenv local 3.10.12
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -e .
python3.10 -m pip install -e ".[qwen]"
```

For Qwen ASR, keep the supported runtime from the training notebook:
`qwen-asr==0.0.6` with `transformers==4.57.6`.

Check the machine:

```bash
python3.10 --version
nvidia-smi
ffmpeg -version
```

## Important Folder Rule

Always pass an input root that contains the year folder.

Good:

```text
/mnt/amc-data/2023/<call_id>/audio.opus
--input /mnt/amc-data
```

For a one-call or one-year test, create a small staging folder that still keeps
the `year/call_id/audio.opus` layout. Do not point `--input` directly at
`/mnt/amc-data/2023`, because then the year level is no longer visible to the
pipeline.

## One Call

Use this for the call you tested earlier:

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

Run a quick discovery check:

```bash
python3.10 -m amc_pipeline.cli dry-run \
  --input "$ONE_IN" \
  --output "$ONE_OUT"
```

Run stage by stage. On one A10 GPU, the safest ASR pattern is one model per
command so the model is unloaded before the next model starts.

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

Listen in Jupyter:

```python
from IPython.display import Audio, display

display(Audio("/tmp/amc-one-output/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus"))
```

## One Folder Or One Year

If the folder already contains year folders, use it directly:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-output

python3.10 -m amc_pipeline.cli dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"
```

If you want only one year, make a staging root with a symlink:

```bash
export YEAR=2023
export SRC_ROOT=/mnt/amc-data
export YEAR_IN=/tmp/amc-year-$YEAR
export YEAR_OUT=/mnt/amc-output-$YEAR

rm -rf "$YEAR_IN"
mkdir -p "$YEAR_IN"
ln -s "$SRC_ROOT/$YEAR" "$YEAR_IN/$YEAR"
```

Then run the same stage commands using `"$YEAR_IN"` and `"$YEAR_OUT"`.

## Full Folder

Use this when you want the full dataset in one run on one machine:

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

For production, stage-by-stage is easier to verify and resume:

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

Rerun the failed stage with the same `--input` and `--output`. State is kept
inside `<output>/.pii_pipeline/`.

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

To pause a long preprocessing run:

```bash
python3.10 -m amc_pipeline.cli pause --output "$AMC_OUT" --global
```

## Output Files

Important outputs:

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

## Multiple A10 GPU Instances

There are two safe ways to use multiple AWS instances with shared storage.

Do not run the exact same stage and same model from multiple instances into the
same SQLite output folder. That causes duplicate work and can create database
locking problems on shared filesystems.

### Option A: Model-Parallel Shared Run

Use this when you have up to four A10 instances and want all instances to work
on the same dataset. One instance owns preprocessing, then each GPU instance
owns exactly one ASR model.

Use a shared Postgres state store, not SQLite.

Install the Postgres client package on every instance:

```bash
python3.10 -m pip install "psycopg[binary]>=3.1"
```

Create `config.shared.yaml` on shared storage:

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

Run discovery and preprocessing once:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted
export CFG=/mnt/amc-data-ebs/AMC/config.shared.yaml

python3.10 -m amc_pipeline.cli --config "$CFG" dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"

python3.10 -m amc_pipeline.cli --config "$CFG" run-stage preprocess \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --vad-backend silero
```

Then run these on different A10 instances at the same time.

Instance 1:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models whisper
```

Instance 2:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models qwen \
  --asr-batch-sizes qwen=1
```

Instance 3:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models cohere \
  --asr-batch-sizes cohere=1
```

Instance 4:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models granite \
  --asr-batch-sizes granite=1
```

After all ASR model jobs finish, run the remaining stages once from one
instance:

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

Rules for this mode:

- Preprocess once.
- Each ASR model runs on exactly one instance.
- Do not run `qwen` on two instances against the same output.
- Run `normalize`, `consensus`, `pii`, `align`, `mask-plan`, `redact`,
  `validate`, and `manifest` once after ASR completes.
- Use Postgres state for shared multi-instance runs.

### Option B: N-Way Sharded Runs

Use this when you have any number of A10 instances and want no shared database
writes during processing. Each instance gets a deterministic subset of call IDs,
its own output folder, and its own pipeline state.

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

On each instance, set a different `SHARD_ID`:

```bash
export SHARD_ID=0
export SHARD_IN=/mnt/amc-shards/shard-$SHARD_ID
export SHARD_OUT=/mnt/amc-output-shards/shard-$SHARD_ID

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

Repeat with `SHARD_ID=1`, `SHARD_ID=2`, etc. Each shard is independent and
resumable.

To collect only final redacted call audio into one folder:

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

Keep the shard output folders if you need full per-shard state, reports,
training segments, and manifests.

## Useful Inspection Commands

Show the final redacted files:

```bash
find "$AMC_OUT" -path '*/audio.*' -type f | head -20
```

Show segment manifests:

```bash
ls -lh "$AMC_OUT/manifests"
head -5 "$AMC_OUT/manifests/all_segments.csv"
```

Show stage reports:

```bash
find "$AMC_OUT/.pii_pipeline/reports" -maxdepth 1 -type f -print
```

Show recorded failures:

```bash
python3.10 - <<'PY'
import sqlite3
import os
from pathlib import Path

db = Path(os.environ["AMC_OUT"]) / ".pii_pipeline/state/pipeline.sqlite3"
conn = sqlite3.connect(db)
for row in conn.execute("select scope, scope_id, error from failures order by created_at desc limit 20"):
    print(row)
PY
```

Progress bars are enabled by default. Add `--no-progress` to any command for
quiet logs.
