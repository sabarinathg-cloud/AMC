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

The single-env install above works for one machine, but the stages actually need
**two isolated environments** (see next section). For any multi-machine / GPU run,
use `ops/setup_env.sh` instead of installing into the base interpreter.

## Environments (two isolated venvs)

The pipeline stages need mutually-incompatible dependency versions, so one environment
cannot hold all of them. `ops/setup_env.sh` builds two virtualenvs and the orchestrator
picks the right one per stage automatically.

| venv | stages | key pins |
| --- | --- | --- |
| `main` | preprocess, asr_whisper, asr_qwen, asr_cohere, asr_granite, normalize, consensus, pii, mask_plan, redact, validate, manifest | `torch 2.5.1+cu121`, `torchvision 0.20.1`, `torchaudio 2.5.1`, `transformers 4.57.6`, `accelerate 1.12.0`, `qwen-asr 0.0.6`, `faster-whisper>=1.1.1`, `ctranslate2>=4.5.0`, `nvidia-cudnn-cu12 9.1.0.70`, `huggingface_hub 0.36.2`, `gliner`, `spacy`, `numpy<2` |
| `align` | align (whisperx forced alignment) | `whisperx 3.4.2`, `pyannote-audio 3.4.0`, `transformers 4.56.2`, `huggingface_hub 0.35.0`, `torch 2.5.1+cu121`, `numpy<2` |

Why two:

- `qwen-asr==0.0.6` hard-pins `transformers==4.57.6` + `accelerate==1.12.0` and is
  documented to require a fresh, isolated environment. Cohere/Granite ASR also need
  transformers 4.57.x.
- `whisperx` (align only) pulls `pyannote-audio` + a different transformers/hf set. The
  current whisperx (3.8.x) demands `torch~=2.8` and `numpy>=2.1`, which conflicts with the
  ASR stack. We pin the last torch-2.5.1-compatible line (`whisperx==3.4.2`,
  `pyannote-audio==3.4.0`, `transformers==4.56.2`, `huggingface_hub==0.35.0`).

Build both venvs (run this once per fleet — the venvs live on shared storage and are
reused by every instance):

```bash
cd /mnt/amc-data/AMC
bash ops/setup_env.sh          # build-if-needed on $AMC_VENV_ROOT (default /mnt/amc-data/venvs), then verify
bash ops/setup_env.sh --verify # re-check imports + CUDA only
FORCE_REBUILD=1 bash ops/setup_env.sh   # rebuild from scratch
```

Key properties:

- **Build once, reuse everywhere.** Both venvs are created under `$AMC_VENV_ROOT`
  (default `/mnt/amc-data/venvs`) on shared NFS, so recreated instances (new IDs, fresh
  local disks) do not reinstall multi-GB wheels — they import from the shared venv.
- **Safe under a boot storm.** A `flock` ensures exactly one builder; other instances wait
  for the ready marker, then each runs the import/CUDA verification locally.
- **Signature-gated rebuilds.** A hash of `requirements-gpu.txt`, `requirements-align.txt`,
  `constraints-gpu.txt`, and `pyproject.toml` keys the `.ready` marker. Change a pin →
  automatic rebuild on the next run; otherwise it is a fast no-op.
- **Self-provisioning.** `ops/resume_shard.sh` (the systemd auto-resume worker) calls
  `setup_env.sh` before claiming any shard, so a fresh instance builds/verifies the env
  automatically. The manual SSM smoke path does not, so run `setup_env.sh` first there.

Override the location with `AMC_VENV_ROOT=/some/shared/path`. To skip env setup in the
resume worker (e.g. for a single-env debug box) set `AMC_SKIP_ENV_SETUP=1`;
`run_shard_no_docker.sh` then falls back to `PYTHON_BIN` for every stage.

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
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models qwen
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models cohere
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --models granite
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
  --models qwen

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models cohere

python3.10 -m amc_pipeline.cli run-stage asr \
  --input "$ONE_IN" \
  --output "$ONE_OUT" \
  --models granite

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
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models qwen
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models cohere
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --models granite
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
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models qwen
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models cohere
python3.10 -m amc_pipeline.cli run-stage asr --input "$AMC_IN" --output "$AMC_OUT" --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models granite
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

### No-Docker SSM Run

Use this mode for the 5-instance A10 fleet when Docker GPU runtime is not
available. Each instance computes its shard index from the current SSM instance
list, writes to a separate shard output folder, and records status under the run
root.

Create a small 2026 review set first. This run root is kept for the full run
later; do not create a separate full-run output if the reviewed calls should be
reused.

```bash
export AWS_REGION=us-east-1
export PROJECT_TAG=amc-ec2-fleet
export RUN_ROOT=/mnt/amc-runs/2026-review-full

ONE_ID=$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[0].InstanceId' \
  --output text)

SETUP_ID=$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "$ONE_ID" \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["cd /mnt/amc-data/AMC","git config --global --add safe.directory /mnt/amc-data/AMC || true","git pull --ff-only","python3.10 ops/create_call_subset.py --source-root /mnt/amc-data --year 2026 --output-root /mnt/amc-runs/2026-review-full/input --limit 100 --mode symlink --force"]}' \
  --query 'Command.CommandId' \
  --output text)

echo "$SETUP_ID"
```

Check setup output:

```bash
aws ssm get-command-invocation \
  --region "$AWS_REGION" \
  --command-id "$SETUP_ID" \
  --instance-id "$ONE_ID" \
  --query '{Status:Status,StdOut:StandardOutputContent,StdErr:StandardErrorContent}' \
  --output json
```

Build the environments fleet-wide (once). This replaces the old ad-hoc root `pip`
install.

IMPORTANT: the repo lives on shared NFS, so do NOT `git pull` from every instance at
once -- concurrent pulls fight over `.git/refs` ("cannot lock ref" / "Cannot
fast-forward to multiple branches"). Pull on ONE instance, then build on all.

First, pull once on a single instance:

```bash
ALL_IDS=($(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'InstanceInformationList[?PingStatus==`Online`].InstanceId' \
  --output text))

ONE_ID=$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[0].InstanceId' \
  --output text)

PULL_ID=$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "$ONE_ID" \
  --document-name AWS-RunShellScript \
  --timeout-seconds 600 \
  --parameters '{"commands":["cd /mnt/amc-data/AMC","git config --global --add safe.directory /mnt/amc-data/AMC || true","git fetch --prune origin","git reset --hard origin/main","ls -l ops/setup_env.sh"]}' \
  --query 'Command.CommandId' \
  --output text)

echo "$PULL_ID"
```

Then build both shared venvs on **all** online instances (no git pull here): the first
one builds under a `flock`; the rest wait for the ready marker and then verify imports +
CUDA locally. It is idempotent, so it is safe to re-send on every deploy.

```bash
ENV_ID=$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "${ALL_IDS[@]}" \
  --document-name AWS-RunShellScript \
  --timeout-seconds 3600 \
  --parameters '{"executionTimeout":["3600"],"commands":["cd /mnt/amc-data/AMC","bash ops/setup_env.sh"]}' \
  --query 'Command.CommandId' \
  --output text)

echo "$ENV_ID"
```

Check the build/verify output per instance (look for `all environments ready and
verified`):

```bash
for id in "${ALL_IDS[@]}"; do
  echo "== $id =="
  aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$ENV_ID" --instance-id "$id" \
    --query '{Status:Status,Out:StandardOutputContent,Err:StandardErrorContent}' --output text | tail -n 20
done
```

Launch the 5-way smoke run:

```bash
export AMC_IN=/mnt/amc-runs/2026-review-full/input
export RUN_ROOT=/mnt/amc-runs/2026-review-full
export STAGES="preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest"

CMD_ID=$(bash ops/ssm_submit_no_docker.sh)
echo "$CMD_ID"
```

Monitor AWS command status and shard-level pipeline progress:

```bash
export RUN_ROOT=/mnt/amc-runs/2026-review-full
bash ops/ssm_status_no_docker.sh "$CMD_ID"
```

Resume the same smoke run after any interruption:

```bash
export AMC_IN=/mnt/amc-runs/2026-review-full/input
export RUN_ROOT=/mnt/amc-runs/2026-review-full
CMD_ID=$(bash ops/ssm_submit_no_docker.sh)
echo "$CMD_ID"
```

Completed stage markers live in:

```bash
/mnt/amc-runs/2026-review-full/outputs/shard-*/.pii_pipeline/stage_markers/
```

Item-level state lives in:

```bash
/mnt/amc-runs/2026-review-full/outputs/shard-*/.pii_pipeline/state/pipeline.sqlite3
```

After the 100-call review set passes, expand the same input folder to the full
2026 set. The existing outputs and SQLite state are not deleted. The stage
markers are tied to an input signature, so the full run re-enters each stage and
item-level resume skips the already processed calls.

```bash
export AWS_REGION=us-east-1
export PROJECT_TAG=amc-ec2-fleet
export RUN_ROOT=/mnt/amc-runs/2026-review-full

ONE_ID=$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'sort_by(InstanceInformationList[?PingStatus==`Online`], &InstanceId)[0].InstanceId' \
  --output text)

SETUP_ID=$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "$ONE_ID" \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["cd /mnt/amc-data/AMC","git config --global --add safe.directory /mnt/amc-data/AMC || true","git pull --ff-only","python3.10 ops/create_call_subset.py --source-root /mnt/amc-data --year 2026 --output-root /mnt/amc-runs/2026-review-full/input --limit 100000 --mode symlink --force"]}' \
  --query 'Command.CommandId' \
  --output text)

echo "$SETUP_ID"
```

Launch the full run:

```bash
export AMC_IN=/mnt/amc-runs/2026-review-full/input
export RUN_ROOT=/mnt/amc-runs/2026-review-full
export STAGES="preprocess asr_whisper asr_qwen asr_cohere asr_granite normalize consensus pii align mask_plan redact validate manifest"

CMD_ID=$(bash ops/ssm_submit_no_docker.sh)
echo "$CMD_ID"
```

### Auto-Resume After Reboot / Spot Reclaim (systemd)

`ssm_submit_no_docker.sh` is one-shot: if an instance reboots, is spot-reclaimed, or is
OOM-killed mid-run, nothing relaunches the pipeline. For long fleet runs, install the
systemd-based auto-resume instead. Each instance then runs its shard to completion and
**automatically continues from where it left off on the next boot**, with no further SSM
submits.

Resume is safe because it is idempotent: `run_shard_no_docker.sh` writes a per-stage
marker only after that stage exits `0`, and every stage skips already-finished work via
the per-shard SQLite state (segments, model results, artifacts, redacted files). So a
restart skips completed stages and, inside an interrupted stage, skips the items that
already finished.

**Instance IDs are not assumed to be stable.** If the fleet is torn down and recreated
(e.g. nightly), every instance comes up with a new instance ID. Shard assignment is
therefore **claim-based**, not pinned to instance identity:

- The unit of work is the *shard index* — a deterministic call-id hash partition whose
  state lives at `$RUN_ROOT/outputs/shard-N` on shared NFS. `NUM_SHARDS` is fixed for the
  run.
- On boot, each instance atomically **claims an incomplete shard** from a lease pool at
  `$RUN_ROOT/shards/` (atomic `mkdir` lock + heartbeat). It runs that shard, then claims
  the next incomplete one, until none remain.
- A lease whose heartbeat is older than `AMC_LEASE_TTL` (default 300s) is considered dead
  and is **stolen** via an atomic rename, so a brand-new instance picks up the shard whose
  previous owner disappeared and resumes that partition's state in place.
- Extra instances (more boxes than incomplete shards) idle and re-scan, ready to take over
  if an owner dies; fewer instances simply process multiple shards in sequence.

```bash
export AWS_REGION=us-east-1
export PROJECT_TAG=amc-ec2-fleet
export AMC_IN=/mnt/amc-runs/2026-review-full/input
export RUN_ROOT=/mnt/amc-runs/2026-review-full
export NUM_SHARDS=5          # fixed data-partition count for this run
# AUTO_PULL=1 makes the single setup instance refresh the shared repo checkout first.
AUTO_PULL=1 bash ops/install_autoresume.sh
```

This writes a durable run config to `$RUN_ROOT/run.env` plus the stable pointer
`/mnt/amc-runs/active.env` (note: **no instance IDs** are stored — claiming is dynamic),
installs `ops/resume_shard.sh` to `/usr/local/bin/amc_resume_shard.sh` and the
`amc-shard.service` unit on every online instance, and `systemctl enable --now`s it. The
unit starts at boot after the network and the shared NFS mount are ready, and is restarted
on failure.

**Make recreated instances self-install.** A recreated instance has a fresh disk, so the
local unit and wrapper are gone. Put `ops/userdata_bootstrap.sh` in the launch template /
launch configuration **user data** so every new instance reinstalls and enables the
service on first boot (it reads the wrapper + unit from the shared repo and is idempotent):

```bash
#!/usr/bin/env bash
# (launch-template user data) — install AMC auto-resume on every new instance
REPO_DIR=/mnt/amc-data/AMC bash /mnt/amc-data/AMC/ops/userdata_bootstrap.sh
```

If you cannot edit user data, just re-run `bash ops/install_autoresume.sh` after each
recreation — it re-pushes the service to whatever instances are currently online.

Monitor (status JSON lives on shared NFS, so one instance shows all shards):

```bash
FIRST_ID=$(echo "$INSTANCE_IDS" | awk '{print $1}')
aws ssm send-command --region "$AWS_REGION" --instance-ids "$FIRST_ID" \
  --document-name AWS-RunShellScript \
  --parameters "{\"commands\":[\"for f in $RUN_ROOT/status/shard-*.json; do cat \$f; echo; done\"]}" \
  --query 'Command.CommandId' --output text
```

Stop the fleet (halts the resume loop on every box without uninstalling):

```bash
# from any instance with the shared mount:
touch "$RUN_ROOT/STOP"
# optionally also: systemctl disable --now amc-shard.service   (per instance)
```

Resume a stopped fleet: `rm -f "$RUN_ROOT/STOP"` then `systemctl start amc-shard.service`
on each instance (or just reboot — the enabled unit starts automatically).

Collect final masked audio after all shards complete:

```bash
export FINAL_OUT=/mnt/amc-redacted-final/2026
mkdir -p "$FINAL_OUT"

for shard_dir in /mnt/amc-runs/2026-review-full/outputs/shard-*; do
  rsync -a \
    --exclude '/.pii_pipeline/' \
    --include '*/' \
    --include 'audio.*' \
    --exclude '*' \
    "$shard_dir"/ "$FINAL_OUT"/
done
```

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
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models qwen
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models cohere
docker/run_stage.sh --config /app/docker/config.docker.yaml run-stage asr --input /data --output /output --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" --discovery-hash-mode path --models granite
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
  --models qwen

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

## ASR Throughput / Batching

The Qwen, Cohere, and Granite adapters use **duration-budgeted dynamic batching**:
segments are sorted by duration and packed into a batch until either a count cap
or an audio-seconds budget is hit. This keeps the GPU busy without the OOM risk of
a fixed large batch (a too-large batch still self-corrects via CUDA-OOM halving).

Do **not** force `--asr-batch-sizes <model>=1` anymore. Batch=1 leaves the GPU
mostly idle. The run scripts no longer set it; the defaults below apply instead.

Per-model `asr_models.<model>` config knobs (all optional, backward compatible):

| Knob | Default | Meaning |
| --- | --- | --- |
| `batch_size` | whisper 32 / qwen 8 / cohere 4 / granite 4 | Count cap fallback (also via `--asr-batch-sizes`). |
| `max_batch_size` | unset | Count cap; overrides `batch_size` when set. |
| `batch_audio_sec_budget` | qwen ~240 / cohere ~160 / granite ~160 | Max summed audio seconds per batch. |
| `dtype` | `auto` (qwen/granite bf16 on CUDA, cohere float32) | `float32` / `float16` / `bfloat16`. |
| `attn_implementation` | `sdpa` (auto, falls back) | Attention kernel; SDPA is faster and numerically equivalent. |
| `prefetch` | `true` | Overlap CPU audio loading with GPU `generate` (Cohere/Granite). |

Example `config.yaml`:

```yaml
asr_models:
  cohere:
    batch_audio_sec_budget: 160
    max_batch_size: 8
    # dtype: bfloat16   # enable ONLY after the parity check below passes
  granite:
    batch_audio_sec_budget: 160
```

### Quality gate before changing precision

`dtype` and batch-size changes must not change transcripts. SDPA and dynamic
batching are output-equivalent and on by default. Cohere defaults to **float32**
(bit-for-bit unchanged). Before enabling `cohere.dtype: bfloat16`, validate on the
GPU host with the parity harness:

```bash
python3 ops/asr_parity_check.py --input "$AMC_IN" --output "$AMC_OUT" \
  --model cohere --limit 50 --check-bf16
```

It reports exact-match rate and WER for batch=1 vs dynamic batching, and float32
vs bf16. Only keep bf16 if exact-match is ~1.0 and WER is negligible.

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
  --models qwen
```

Machine 3:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models cohere
```

Machine 4:

```bash
python3.10 -m amc_pipeline.cli --config "$CFG" run-stage asr \
  --input "$AMC_IN" \
  --output "$AMC_OUT" \
  --models granite
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
