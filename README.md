# AMC PII Audio Pipeline

Resumable audio preprocessing, ASR, normalization, transcript PII detection,
forced alignment, channel-specific masking, validation, and manifest generation.

## One-File Smoke Test

Prepare a single-call input folder. The call ID is the parent directory name.

```bash
mkdir -p /tmp/amc-one/<year>/<call_id>
cp /mnt/amc-data/<year>/<call_id>/audio.opus /tmp/amc-one/<year>/<call_id>/audio.opus
rm -rf /tmp/amc-one-output
```

Run each stage independently:

```bash
python3 -m amc_pipeline.cli dry-run --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage preprocess --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage asr --input /tmp/amc-one --output /tmp/amc-one-output --models whisper,qwen,cohere,granite
python3 -m amc_pipeline.cli run-stage normalize --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage consensus --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage pii --input /tmp/amc-one --output /tmp/amc-one-output --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
python3 -m amc_pipeline.cli run-stage align --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage mask-plan --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage redact --input /tmp/amc-one --output /tmp/amc-one-output --mask-strategy beep --allow-fallback-format wav
python3 -m amc_pipeline.cli run-stage validate --input /tmp/amc-one --output /tmp/amc-one-output
python3 -m amc_pipeline.cli run-stage manifest --input /tmp/amc-one --output /tmp/amc-one-output
```

Progress bars are enabled by default. Add `--no-progress` to any command if you
want quiet logs.

The final redacted audio mirrors the input structure under `/tmp/amc-one-output`.
Training segments and transcripts are under `/tmp/amc-one-output/<year>/segments/`
and manifests are under `/tmp/amc-one-output/manifests/`.

## Full Run

```bash
python3 -m amc_pipeline.cli run \
  --input /mnt/amc-data \
  --output /mnt/amc-redacted \
  --models whisper,qwen,cohere,granite \
  --detectors regex,gliner,piiranha,spacy,rule_name,saved_json
```
