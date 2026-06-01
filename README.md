# AMC PII Audio Pipeline

Pipeline for AMC audio preprocessing, ASR, normalization, PII
detection, force alignment, channel-specific audio masking, validation, and ASR
training manifests.

The main goal is simple:

- keep the original folder structure
- process each channel separately
- detect transcript-based PII
- align the detected PII back to audio time
- mask only the affected channel and time range
- keep unmasked mono channel segments for ASR training
- keep enough state to resume after failures

Example input:

```text
/mnt/amc-data/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

Example redacted output:

```text
<output>/2023/aaf5285a-3d70-56a8-87e3-77c26d547494/audio.opus
```

The pipeline state, cache, and reports will be inside:

```text
<output>/.pii_pipeline/
```

Use Python 3.10.12 for this project.

## Install

```bash
cd /mnt/amc-data-ebs/AMC
git pull

pyenv local 3.10.12
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -e .
python3.10 -m pip install -e ".[qwen]"
```

For Qwen ASR, keep the same supported runtime used in the training notebook:

```text
qwen-asr==0.0.6
transformers==4.57.6
```

Check the machine:

```bash
python3.10 --version
nvidia-smi
ffmpeg -version
```

## Docker GPU Runtime

Docker is supported for the GPU runtime. Model weights are not copied into the
image. The model directory is mounted at runtime so the image stays small and
the same image can be used on all A10 machines.

Host requirements:

```bash
nvidia-smi
docker --version
```

The EC2 host must have NVIDIA Container Toolkit installed. GPU access is tested
by running containers with `--gpus all`.

Build the image:

```bash
cd /mnt/amc-data-ebs/AMC
docker/build_gpu.sh
```

Test GPU access inside the container:

```bash
docker/test_gpu.sh
```

Expected test result:

```text
cuda_available: True
cuda_device_count: 1
cuda_device_name: NVIDIA A10G
```

Default Docker mount layout:

```text
/data    -> input audio root
/output  -> pipeline output root
/models  -> local model snapshots
/cache   -> Hugging Face and Torch cache
```

Run one stage through Docker:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted
export MODEL_ROOT=/mnt/amc-data/pipeline/models
export CACHE_ROOT=/mnt/amc-cache

docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage preprocess \
  --input /data \
  --output /output \
  --vad-backend silero
```

Run ASR model by model through Docker:

```bash
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models whisper
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models qwen --asr-batch-sizes qwen=1
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models cohere --asr-batch-sizes cohere=1
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models granite --asr-batch-sizes granite=1
```

Run the known one-call test through Docker:

```bash
export SRC_ROOT=/mnt/amc-data
export MODEL_ROOT=/mnt/amc-data/pipeline/models
export CACHE_ROOT=/mnt/amc-cache
export YEAR=2023
export CALL_ID=aaf5285a-3d70-56a8-87e3-77c26d547494

docker/run_one_call.sh
```

Docker Compose can also be used:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted
export AMC_MODEL_ROOT=/mnt/amc-data/pipeline/models
export AMC_CACHE_ROOT=/mnt/amc-cache

docker compose -f docker/compose.gpu.yaml build
docker compose -f docker/compose.gpu.yaml run --rm amc --help
```

For multi-machine Docker runs, use the same rules as the normal runtime:

- one shared Postgres state store for a shared model-parallel run
- or one independent sharded output folder per machine
- do not run the same model and same stage from multiple machines into the same
  SQLite output folder

## Folder Rule

Always pass an input root that contains the year folder.

Correct:

```text
/mnt/amc-data/2023/<call_id>/audio.opus
--input /mnt/amc-data
```

For one-call or one-year testing, create a small staging folder that still has
the same `year/call_id/audio.opus` layout.

Do not point `--input` directly at `/mnt/amc-data/2023`, because then the
pipeline cannot see the year level correctly.

## One Call Test

Command set for testing one call end to end.

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

First check that discovery works:

```bash
python3.10 -m amc_pipeline.cli dry-run \
  --input "$ONE_IN" \
  --output "$ONE_OUT"
```

Then run the stages one by one. On one A10 GPU, run one ASR model
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

If the folder already contains year folders, use it directly:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-output

python3.10 -m amc_pipeline.cli dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"
```

To process only one year, make a staging root with a symlink:

```bash
export YEAR=2023
export SRC_ROOT=/mnt/amc-data
export YEAR_IN=/tmp/amc-year-$YEAR
export YEAR_OUT=/mnt/amc-output-$YEAR

rm -rf "$YEAR_IN"
mkdir -p "$YEAR_IN"
ln -s "$SRC_ROOT/$YEAR" "$YEAR_IN/$YEAR"
```

Then use `"$YEAR_IN"` as `--input` and `"$YEAR_OUT"` as `--output`.

## Full Folder On One Machine

To run the full dataset on one machine in one command:

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

For production, stage by stage is easier to verify and
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

## Recommended Large-Scale Mode

For a large number of calls on N A10 machines with shared storage, the fastest
safe mode is deterministic call sharding:

- all machines read the same input root
- each machine gets a different `--shard-index`
- each machine writes to its own output folder
- each shard runs all four ASR models for maximum accuracy
- `--discovery-hash-mode path` avoids reading every audio file during discovery
- no shared SQLite writes happen during processing

This keeps the same accuracy as the single-machine full run because every shard
still runs Whisper, Qwen, Cohere, Granite, normalization, consensus, PII,
WhisperX alignment, and redaction.

Machine 0 of 4:

```bash
export AMC_IN=/mnt/amc-data
export NUM_SHARDS=4
export SHARD_INDEX=0
export AMC_OUT=/mnt/amc-output-shards/shard-$SHARD_INDEX

python3.10 -m amc_pipeline.cli run-stage preprocess --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --vad-backend silero
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models whisper
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models qwen --asr-batch-sizes qwen=1
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models cohere --asr-batch-sizes cohere=1
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models granite --asr-batch-sizes granite=1
python3.10 -m amc_pipeline.cli run-stage normalize --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
python3.10 -m amc_pipeline.cli run-stage consensus --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
python3.10 -m amc_pipeline.cli run-stage pii --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
python3.10 -m amc_pipeline.cli run-stage align --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
python3.10 -m amc_pipeline.cli run-stage mask-plan --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
python3.10 -m amc_pipeline.cli run-stage redact --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --mask-strategy beep --allow-fallback-format wav
python3.10 -m amc_pipeline.cli run-stage validate --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
python3.10 -m amc_pipeline.cli run-stage manifest --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
```

For the other machines, only change `SHARD_INDEX`:

```bash
export SHARD_INDEX=1
export SHARD_INDEX=2
export SHARD_INDEX=3
```

For N machines, set `NUM_SHARDS=N` and use shard indexes from `0` to `N - 1`.

Docker version of the same shard run:

```bash
export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-output-shards/shard-0
export MODEL_ROOT=/mnt/amc-data/pipeline/models
export CACHE_ROOT=/mnt/amc-cache
export NUM_SHARDS=4
export SHARD_INDEX=0

docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage preprocess --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --vad-backend silero
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models whisper
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models qwen --asr-batch-sizes qwen=1
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models cohere --asr-batch-sizes cohere=1
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models granite --asr-batch-sizes granite=1
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage normalize --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage consensus --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage pii --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage align --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage mask-plan --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage redact --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --mask-strategy beep --allow-fallback-format wav
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage validate --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage manifest --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path
```

Collect final redacted audio from shard output folders:

```bash
export FINAL_OUT=/mnt/amc-redacted-final
mkdir -p "$FINAL_OUT"

for shard_dir in /mnt/amc-output-shards/shard-*; do
  rsync -a \
    --exclude '/.pii_pipeline/' \
    --include '*/' \
    --include 'audio.*' \
    --exclude '*' \
    "$shard_dir"/ "$FINAL_OUT"/
done
```

## Resume After Failure

If anything fails, rerun the failed stage with the same `--input` and
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

To pause a long preprocessing run:

```bash
python3.10 -m amc_pipeline.cli pause --output "$AMC_OUT" --global
```

## Outputs To Check

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

Use one of these two patterns. Do not run the exact same stage and same
model from multiple machines into the same SQLite output folder.

That unsafe pattern can duplicate work and cause database locking issues on
shared storage.

### Option A: One Shared Run, One ASR Model Per Machine

Use this when up to four A10 machines should work on
the same dataset.

The rule is:

- preprocess once
- run one ASR model per machine
- wait until all ASR jobs finish
- run the remaining stages once

For this shared run, use Postgres state, not SQLite.

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

Set this on every machine:

```bash
cd /mnt/amc-data-ebs/AMC
git pull

export AMC_IN=/mnt/amc-data
export AMC_OUT=/mnt/amc-redacted
export CFG=/mnt/amc-data-ebs/AMC/config.shared.yaml
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Run discovery and preprocessing once from one machine:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" dry-run \
  --input "$AMC_IN" \
  --output "$AMC_OUT"

python3.10 -m amc_pipeline.cli --config "$CFG" run-stage preprocess \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --vad-backend silero
```

Then run these in parallel, one command per A10 machine.

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

After all four ASR jobs finish, run the rest once from one machine:

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

- Run preprocessing once.
- Run each ASR model on exactly one machine.
- Do not run `qwen` on two machines against the same output.
- Run post-ASR stages only once after all ASR jobs finish.
- Use Postgres state for this shared multi-machine mode.

### Option B: N-Way Sharded Runs

Use this when there are many A10 machines or when the safest parallel
setup. Each machine gets its own subset of call IDs, its own output folder, and
its own state database.

This avoids shared database writes during processing.

Create call-ID shards on shared storage:

```bash
export AMC_IN=/mnt/amc-data
export SHARD_ROOT=/mnt/amc-shards
export N=4

rm -rf "$SHARD_ROOT"
for shard_idx in $(seq 0 $((N - 1))); do
  mkdir -p "$SHARD_ROOT/shard-$shard_idx"
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

Use a different `SHARD_ID` on each machine.

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

Then repeat with `SHARD_ID=1`, `SHARD_ID=2`, etc.

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

Keep the shard output folders if full state, reports, training
segments, and manifests from each shard.

## Commands To Inspect Output

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

Progress bars are enabled by default. For quieter logs, add
`--no-progress` to the command.
