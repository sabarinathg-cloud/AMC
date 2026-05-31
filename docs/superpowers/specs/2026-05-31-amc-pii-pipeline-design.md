# AMC Audio PII Pipeline Design

Date: 2026-05-31
Status: Approved design draft
CLI command: `amc-pipeline`

## Goal

Build a production-grade, local-first audio PII redaction and ASR training data pipeline from the existing experiments in:

- `/Users/gsabarinath/amc/training_2026_full_optimized_complete(1)(1).py`
- `/Users/gsabarinath/amc/pii.py`

The new implementation will be a clean Python 3.11 package and CLI. The experimental notebooks/scripts remain untouched and serve as references for model integrations and current logic.

The pipeline has two first-class outputs from day one:

1. Shareable redacted full audio that mirrors the input folder structure and preserves audio integrity.
2. ASR training artifacts: unmasked mono per-channel segment audio plus complete manifests containing transcripts, normalized transcripts, final transcript, PII metadata, and provenance.

## Non-Negotiable Requirements

- Never modify originals.
- Require a separate output folder.
- Preserve full redacted audio duration, timing, channel count, channel layout, and original format whenever possible.
- Preserve non-speech and non-PII audio unchanged.
- Process stereo/multichannel audio channel-by-channel for ASR, PII, alignment, and masking.
- Mask only the affected channel at the affected interval.
- Keep all four ASR model outputs, normalized model outputs, final consensus transcript, and best-effort redaction transcript when needed.
- Run all four ASR models on every supported-language speech segment by default.
- Use final consensus transcript for PII by default; make all-model-transcript PII configurable.
- Require forced alignment for English and Spanish.
- If required alignment fails for a PII-relevant segment, fail the segment/call for redacted output and make it retryable.
- Default to maximum-recall PII masking.
- Default mask strategy is beep; silence and noise are configurable.
- Make every expensive stage resumable and cacheable.
- Support single-machine SQLite state and distributed Postgres state for multiple AWS instances with shared storage.

## Supported Input Layout

Primary dataset layout:

```text
/mnt/amc-data/
  2022/
    <call_id>/
      audio.opus
  2023/
    <call_id>/
      audio.opus
```

For files named `audio.*`, `call_id` is the parent folder name. The pipeline also recursively processes any supported audio file. For non-`audio.*` files, a stable derived call ID is generated from the relative path.

Supported extensions include:

- `.wav`
- `.mp3`
- `.opus`
- `.ogg`
- `.flac`
- `.m4a`
- `.aac`
- `.wma`
- `.webm`
- any additional formats supported by the reference experiments and FFmpeg configuration

## Output Layout

For input:

```text
/mnt/amc-data/2022/<call_id>/audio.opus
```

Expected outputs:

```text
<output>/
  2022/
    <call_id>/
      audio.opus
    segments/
      <call_id>/
        <segment_id>.wav
    manifests/
      segments.parquet
      segments.csv
      review.xlsx
  manifests/
    all_segments.parquet
    all_segments.csv
    review.xlsx
  .pii_pipeline/
    state/
    cache/
      metadata/
      preprocess/
      asr/
        whisper/
        qwen/
        cohere/
        granite/
      normalized/
      consensus/
      pii/
      alignment/
      masking_plan/
    logs/
    reports/
    temp/
```

If original-format export fails, the pipeline writes a WAV fallback:

```text
<output>/2022/<call_id>/audio.wav
```

The file is marked `completed_with_fallback`, with metadata recording the failed original-format export and allowing later retry.

## Stage Architecture

The pipeline is stage-based. Stages can run end-to-end or independently.

### 1. Discovery

Responsibilities:

- Recursively discover supported audio files.
- Normalize paths relative to input root.
- Infer year and call ID.
- Compute content hash.
- Detect duplicate audio by hash.
- Read common sidecars such as `metadata.json` when present.
- Generate file rows and initial work queue.

Duplicate behavior:

- Expensive processing is reused by content hash.
- Outputs and manifest rows are still emitted for every original path.

### 2. Inspection

Responsibilities:

- Use FFprobe or equivalent to collect codec, container, duration, sample rate, channel count, channel layout, bitrate, stream metadata, and readability.
- Store inspection metadata in state and cache.
- Validate supported audio streams before preprocessing.

### 3. Preprocessing And Segmentation

Responsibilities:

- Decode full audio for working use without modifying originals.
- Create canonical 16 kHz mono channel files for ASR and forced alignment.
- Run VAD per channel.
- Segment speech by pauses and avoid cutting through speech where possible.
- Default max segment length is around 30 seconds.
- Export unmasked mono per-channel segment WAV files for internal ASR training.
- Validate segment duration sample-accurately against planned `start_sample` and `end_sample`.

The canonical working audio can differ from the original. Final redaction is applied to decoded full-channel audio and exported back to the original format or WAV fallback.

Training segments are not the shareable redacted audio. They are internal, unmasked, per-channel clips with PII metadata in the manifests. The shareable deliverable is the mirrored full-call redacted audio under the year/call folder.

### 4. ASR

Default ASR behavior:

- Run all four models on every speech segment:
  - Whisper large v3
  - Qwen3 ASR 1.7B
  - Cohere Transcribe 03-2026
  - Granite 4.0 1B Speech
- Allow YAML and CLI model enable/disable.
- Batch segments across files for GPU throughput.
- Track per-segment, per-model results and failures.
- If one model fails, continue with the others and make that model result retryable.

Whisper additionally performs language detection per segment/channel.

Supported v1 languages:

- English
- Spanish

Unsupported language segments are skipped with a clear reason and excluded from downstream ASR/PII/alignment/redaction.

Each ASR model result stores:

- raw transcript
- timestamps and confidences when available
- language metadata when available
- model path/version/checksum
- runtime metadata
- error/retry info when failed

### 5. Normalization

Responsibilities:

- Normalize every model transcript.
- Preserve mappings needed for PII span handling.
- Handle casing, whitespace, punctuation cleanup, unicode normalization, token cleanup, and number normalization where appropriate.
- Store normalized transcript per model, not only the final transcript.

### 6. Consensus And Best-Effort Transcript

Default consensus:

- Strong consensus requires at least 3 successful ASR models out of 4.
- Use normalized transcripts for agreement scoring.
- Store consensus score and method.

Weak consensus:

- Still create a best-effort redaction transcript.
- Choose best available transcript using a quality score plus configurable model priority.
- Quality score considers model confidence, transcript length sanity, language confidence, alignment readiness, and model-specific priority.
- Mark weak consensus segments for training metadata.

Persist:

- raw transcript per model
- normalized transcript per model
- final consensus transcript
- best-effort redaction transcript where used
- consensus status and quality metrics

### 7. PII Detection

Default behavior:

- Run PII detection on final consensus transcript.
- If consensus is weak, run on the selected best-effort redaction transcript.
- Config can enable PII over every model transcript and union the results.
- Use maximum-recall policy.

PII detectors from the experiments:

- regex/rules baseline, enabled by default
- GLiNER
- Piiranha/token-classification detector
- rule-based person name detector
- spaCy fallback
- saved/existing JSON PII when available

PII categories:

- direct identifiers such as names, phone numbers, email addresses, addresses, DOB, account numbers, card numbers, IDs, passport numbers, SSNs, customer identifiers
- PHI/healthcare entities such as medical condition, medication, lab result, doctor name, institution name, insurance ID, medical code, and related configured entities

PII text is stored internally. Reports can be generated in a redacted/privacy-safe mode that hides raw PII text.

Regex/rules remain enabled by default as a safety baseline, with explicit config/CLI to disable when required.

### 8. Forced Alignment

Responsibilities:

- Use local-only forced alignment for English and Spanish.
- Map PII text spans to word timestamps.
- Alignment is required for PII-relevant segments.
- If required alignment fails for a PII-relevant segment, mark segment/call failed and retryable.
- Do not conservatively mask whole segments by default.

Default timestamp padding:

- `pre_padding_ms: 80`
- `post_padding_ms: 120`

Mask intervals are clipped to audio duration.

### 9. Mask Plan

Responsibilities:

- Convert aligned PII spans to channel-specific intervals.
- Merge overlapping and adjacent intervals.
- Preserve source PII entities, detector sources, confidence, transcript source, and alignment source.
- Store one mask plan per call and channel.

### 10. Redaction

Responsibilities:

- Decode original full audio into arrays for precise channel-specific masking.
- Apply mask only to planned channel/time intervals.
- Default mask is beep.
- Silence and noise are configurable.
- Use short fades to avoid clicks/pops.
- Preserve all non-masked audio samples as unchanged as the codec path allows.
- Export original format first.
- If original-format export fails, export WAV fallback and mark `completed_with_fallback`.
- Preserve metadata/tags when FFmpeg can do so safely.

### 11. Validation

Before completion, validate:

- output is readable
- duration matches original exactly where sample-addressable
- lossy container duration differs by at most 10 ms
- channel count matches original
- channel layout is preserved where available
- sample rate is preserved where possible
- format matches original or fallback is correctly recorded
- planned mask intervals are valid and in range
- segment audio duration matches planned samples

### 12. Manifests And Reports

Complete source of truth:

- Parquet
- CSV

Review-friendly output:

- Excel workbook when practical
- For huge runs, Excel is summarized/partitioned instead of being the only complete artifact

Manifests are both global and per-year.

Segment-level manifest includes:

- `segment_id`
- `call_id`
- `year`
- `source_path_abs`
- `source_path_rel`
- `redacted_audio_path_abs`
- `redacted_audio_path_rel`
- `segment_audio_path_abs`
- `segment_audio_path_rel`
- `channel`
- `start_sec`
- `end_sec`
- `start_sample`
- `end_sample`
- `duration_sec`
- `duration_samples`
- `language`
- `language_confidence`
- raw transcript for each ASR model
- normalized transcript for each ASR model
- final transcript
- best-effort transcript, if used
- consensus method and score
- PII labels, spans, detector sources, confidence, and timestamp ranges
- mask intervals
- alignment status
- validation status
- model versions/checksums
- config hash
- sidecar metadata fields where present

Summary reports include:

- total files
- completed files
- failed files
- skipped unsupported-language segments/files
- fallback outputs
- resumed/skipped cached work
- processing times
- resource usage
- validation/audit results

## State, Cache, And Resumability

Single-machine mode:

- SQLite inside `<output>/.pii_pipeline/state/`

Distributed shared-storage mode:

- Postgres for state, leases, locks, and pause controls
- Shared storage for artifacts

State granularity:

- run
- worker
- file
- audio stream
- segment
- ASR model result
- normalized transcript
- consensus result
- PII detector result
- alignment result
- mask plan
- redacted output
- validation result
- artifact
- lease
- retry
- failure
- pause request

Resume behavior:

- Rerunning the same command or `resume` continues from the last successful stage/model/artifact.
- Completed valid cached artifacts are reused.
- Failed model results, segments, files, or stages can be retried without restarting the full dataset.
- Atomic writes are required: write temp, validate, rename, then mark complete.

Pause behavior:

- Cooperative pause is required.
- Worker-level pause stops a specific worker after its current safe unit.
- Global pause stops all workers for a run.
- Workers save state and release leases before stopping.

## Resource Management

The pipeline detects:

- CPU cores
- RAM
- GPU count
- GPU memory
- disk space

Supported execution:

- CPU-only
- one GPU
- multiple GPUs on one host
- multiple AWS instances with one GPU each, shared storage, and Postgres state

Scheduling:

- Batch across segments and files for GPU throughput.
- Stage/model-level leases allow separate machines to run separate stages or models.
- Workers dynamically size batch/work units based on memory and configured limits.
- Preflight fails if an enabled model path or required dependency is missing.

## CLI Design

Primary commands:

```bash
amc-pipeline dry-run --input /mnt/amc-data --output /mnt/redacted
amc-pipeline run --input /mnt/amc-data --output /mnt/redacted
amc-pipeline resume --output /mnt/redacted
amc-pipeline run-stage preprocess --output /mnt/redacted
amc-pipeline run-stage asr --models whisper,qwen --output /mnt/redacted
amc-pipeline run-stage asr:whisper --output /mnt/redacted
amc-pipeline run-stage pii --detectors gliner,piiranha,regex --output /mnt/redacted
amc-pipeline run-stage align --output /mnt/redacted
amc-pipeline run-stage redact --output /mnt/redacted
amc-pipeline retry-failed --output /mnt/redacted
amc-pipeline validate --output /mnt/redacted
amc-pipeline audit --output /mnt/redacted
amc-pipeline pause --worker-id <id> --output /mnt/redacted
amc-pipeline pause --run-id <id> --global --output /mnt/redacted
amc-pipeline clean-cache --output /mnt/redacted
```

Configuration:

- YAML config is primary.
- CLI flags override YAML.
- Config includes model paths, enabled ASR models, enabled PII detectors, language policy, thresholds, VAD segmentation, masking, padding, workers, GPU allocation, state backend, distributed settings, sidecar handling, fallback behavior, report mode, and cache policy.

## Proposed Package Structure

```text
amc_pipeline/
  __init__.py
  cli/
  config/
  discovery/
  inspection/
  preprocessing/
  segmentation/
  transcription/
    whisper.py
    qwen.py
    cohere.py
    granite.py
  normalization/
  consensus/
  pii/
    regex_rules.py
    gliner.py
    piiranha.py
    spacy_fallback.py
    saved_json.py
  alignment/
  masking/
  audio_io/
  validation/
  state/
    sqlite.py
    postgres.py
    leases.py
  cache/
  reporting/
  manifests/
  resources/
  workers/
  tests/
```

## Testing Strategy

Use small generated fixtures so tests are fast and deterministic.

Required tests:

- discovery preserves relative structure
- `audio.*` call ID comes from parent folder
- duplicate hash reuses processing but emits per-path outputs
- stereo input produces stereo full redacted output
- channel-specific masking affects only the intended channel
- final redacted duration matches original within configured tolerance
- segment WAV duration matches planned samples
- stage resume skips completed artifacts
- failed ASR model result can be retried independently
- required alignment failure marks segment/call retryable
- WAV fallback is generated and marked when original-format export fails
- manifests include raw transcripts, normalized transcripts, final transcript, PII metadata, and paths
- unsupported language is skipped with reason
- pause request stops cooperatively after current safe unit

## Open Implementation Choices

These are design decisions to finalize during implementation planning:

- Select the exact local forced aligner and model artifacts for English and Spanish.
- Decide whether to use SQLAlchemy, SQLModel, or direct SQL for state access.
- Decide exact CLI framework, likely Typer.
- Decide artifact serialization format for intermediate spans and alignments, likely JSONL/Parquet depending on size.
- Decide exact beep/noise generation defaults after listening tests.

## Approved User Decisions

- Both redaction and ASR training outputs are first-class.
- Channel-isolated processing and masking is required.
- Run all four ASR models on every speech segment by default.
- ASR and PII model selection must be configurable.
- PII defaults to maximum recall.
- PII runs on final consensus transcript by default.
- Keep all model transcripts, normalized transcripts, and final transcript.
- Weak consensus uses best available transcript.
- Training segments are mono per channel.
- Full redacted audio preserves original multichannel layout.
- VAD segmentation per channel, max around 30 seconds, avoid cutting speech.
- Forced alignment is required.
- Alignment failure should fail/retry, not conservatively mask.
- Local-only model execution by default.
- English and Spanish are mandatory v1 languages.
- Unsupported languages are skipped with reason.
- SQLite local mode and Postgres distributed mode.
- Stage/model-level distributed leases are required.
- Separate output folder is required.
- Duplicate files compute once but emit outputs per path.
- CLI first, Python API second.
- YAML config with CLI overrides.
- Exact/sample duration validation where possible, max 10 ms lossy tolerance.
- Pre-padding 80 ms, post-padding 120 ms.
- Strong consensus requires 3 of 4 models.
- Reports include Parquet/CSV/Excel with detailed segment rows.
- No dashboard in v1.
- No special encryption/security controls in v1.
- Cooperative pause is required, including worker-level and global pause.
- Audio-only output by default, optional sidecar copying.
- Sidecar metadata should be included in manifests/state.
- Store raw PII text internally; allow redacted report mode.
