# AMC PII Audio Pipeline

Resumable audio preprocessing, ASR, normalization, transcript PII detection,
forced alignment, channel-specific masking, validation, and manifest generation.

Use Python 3.10.12 for the runtime environment.

```bash
pyenv local 3.10.12
python3.10 -m pip install -e .
```

For Qwen ASR, install the same supported runtime used by the training notebook.
`qwen-asr` currently requires `transformers==4.57.6`; newer Transformers
versions can load the package but fail at model initialization.

```bash
python3.10 -m pip install -e ".[qwen]"
```

## One-File Smoke Test

Prepare a single-call input folder. The call ID is the parent directory name.

```bash
mkdir -p /tmp/amc-one/<year>/<call_id>
cp /mnt/amc-data/<year>/<call_id>/audio.opus /tmp/amc-one/<year>/<call_id>/audio.opus
rm -rf /tmp/amc-one-output
```

Run each stage independently:

```bash
python3.10 -m amc_pipeline.cli dry-run --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage preprocess --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage asr --input /tmp/amc-one --output /tmp/amc-one-output --models whisper,qwen,cohere,granite
python3.10 -m amc_pipeline.cli run-stage normalize --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage consensus --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage pii --input /tmp/amc-one --output /tmp/amc-one-output --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
python3.10 -m amc_pipeline.cli run-stage align --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage mask-plan --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage redact --input /tmp/amc-one --output /tmp/amc-one-output --mask-strategy beep --allow-fallback-format wav
python3.10 -m amc_pipeline.cli run-stage validate --input /tmp/amc-one --output /tmp/amc-one-output
python3.10 -m amc_pipeline.cli run-stage manifest --input /tmp/amc-one --output /tmp/amc-one-output
```

Progress bars are enabled by default. Add `--no-progress` to any command if you
want quiet logs.

If the OS kills a heavy ASR run, lower batch sizes or run one model at a time:

```bash
python3.10 -m amc_pipeline.cli run-stage asr \
  --input /tmp/amc-one \
  --output /tmp/amc-one-output \
  --models qwen,cohere,granite \
  --asr-batch-sizes qwen=1,cohere=1,granite=1
```

The final redacted audio mirrors the input structure under `/tmp/amc-one-output`.
Training segments and transcripts are under `/tmp/amc-one-output/<year>/segments/`
and manifests are under `/tmp/amc-one-output/manifests/`.

## Full Run

```bash
python3.10 -m amc_pipeline.cli run \
  --input /mnt/amc-data \
  --output /mnt/amc-redacted \
  --models whisper,qwen,cohere,granite \
  --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
```
