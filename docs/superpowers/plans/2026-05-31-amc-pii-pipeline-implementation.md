# AMC PII Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `amc-pipeline` Python package and CLI for resumable folder-level audio PII redaction plus ASR training segment generation.

**Architecture:** Implement a staged pipeline with durable state, deterministic artifacts, optional heavy model integrations, and a standard-library testable core. The core must run in this bare workspace; production ASR/PII/audio backends activate only when local dependencies, FFmpeg, and model paths are available.

**Tech Stack:** Python 3.11, argparse, sqlite3, json, wave/audioop-compatible PCM handling, subprocess FFprobe/FFmpeg integration when present, optional faster-whisper/qwen-asr/transformers/GLiNER/spaCy/Piiranha.

---

## File Map

- Create `pyproject.toml`: package metadata, console script, pytest config.
- Create `amc_pipeline/config.py`: dataclass config, YAML-or-JSON loading, CLI overrides, experiment-derived default model paths.
- Create `amc_pipeline/models.py`: shared dataclasses for files, segments, ASR results, PII spans, alignments, mask intervals, validations.
- Create `amc_pipeline/state.py`: SQLite state schema, upserts, stage status, leases, pause flags, retries.
- Create `amc_pipeline/discovery.py`: recursive audio discovery, hashing, call/year inference, sidecar metadata.
- Create `amc_pipeline/inspection.py`: FFprobe inspection with WAV fallback inspection for tests.
- Create `amc_pipeline/audio.py`: WAV PCM read/write, FFmpeg decode/encode helpers, beep/silence/noise masking.
- Create `amc_pipeline/segmentation.py`: energy VAD fallback and smart pause-aware chunking.
- Create `amc_pipeline/preprocessing.py`: per-channel canonical segment export and state/artifact persistence.
- Create `amc_pipeline/transcription.py`: model adapter interfaces, local preflight, adapters for Whisper/Qwen/Cohere/Granite, fixture adapter for tests.
- Create `amc_pipeline/normalization.py`: transcript cleanup and number/token normalization.
- Create `amc_pipeline/consensus.py`: normalized transcript agreement and best-effort transcript selection.
- Create `amc_pipeline/pii_detection.py`: regex/rules detector plus optional GLiNER/Piiranha/spaCy adapters.
- Create `amc_pipeline/alignment.py`: required alignment interface, exact transcript-token fallback for tests, optional local model placeholder with strict preflight.
- Create `amc_pipeline/masking.py`: PII-to-time mapping, interval merging, mask plan generation, full-call redaction.
- Create `amc_pipeline/validation.py`: duration/channel/readability/fallback validation.
- Create `amc_pipeline/manifests.py`: CSV/JSONL manifest writers and optional Parquet/Excel writers if pandas is installed.
- Create `amc_pipeline/resources.py`: CPU/RAM/GPU/disk detection without hard dependencies.
- Create `amc_pipeline/pipeline.py`: stage orchestration, run/resume/run-stage/retry/dry-run/audit/pause.
- Create `amc_pipeline/cli.py`: `amc-pipeline` command using argparse.
- Create `tests/`: generated WAV fixture tests for discovery, segmentation, masking, state resume, manifests, CLI smoke.

## Task 1: Package Skeleton And Configuration

- [ ] Create package directories and `pyproject.toml`.
- [ ] Add `PipelineConfig` with defaults from experiment files:
  - model root `/mnt/amc-data/pipeline/models`
  - Whisper `whisper-large-v3`
  - Qwen `qwen3-asr-1.7b`
  - Cohere `cohere-transcribe-03-2026`
  - Granite `granite-4.0-1b-speech`
  - GLiNER `knowledgator/gliner-pii-large-v1.0`
  - Piiranha `iiiorg/piiranha-v1-detect-personal-information`
  - target sample rate `16000`
  - max segment seconds `30`
  - beep mask default
  - English/Spanish support
- [ ] Add config hash and output path helpers.
- [ ] Run `python3 -m py_compile amc_pipeline/config.py`.

## Task 2: Shared Models And SQLite State

- [ ] Define dataclasses for file records, segment records, model results, PII spans, mask intervals, and validation results.
- [ ] Implement SQLite schema with rows for files, segments, model results, artifacts, failures, leases, retries, pause requests, and run metadata.
- [ ] Add idempotent upserts so reruns do not duplicate records.
- [ ] Add cooperative pause flags and stage status transitions.
- [ ] Add tests for resume-safe upsert and pause flags.

## Task 3: Discovery And Inspection

- [ ] Implement recursive discovery over supported audio extensions.
- [ ] Infer `year` from first relative path component and `call_id` from parent folder for `audio.*`.
- [ ] Hash files and group duplicates by content hash.
- [ ] Read and flatten `metadata.json` sidecars when present.
- [ ] Implement FFprobe inspection and standard-library WAV inspection fallback.
- [ ] Add tests for structure preservation, duplicate detection, sidecar ingestion, and WAV inspection.

## Task 4: Audio IO, VAD, Segmentation, And Preprocessing

- [ ] Implement standard-library PCM WAV read/write for mono/stereo fixtures.
- [ ] Implement FFmpeg decode/encode wrappers that fail clearly when FFmpeg is missing.
- [ ] Implement energy VAD fallback per channel.
- [ ] Implement pause-aware chunk splitting with max duration around 30 seconds.
- [ ] Export unmasked mono per-channel segment WAVs under `<output>/<year>/segments/<call_id>/`.
- [ ] Add segment duration validation against planned sample counts.
- [ ] Add tests proving stereo channel splitting and segment sample durations.

## Task 5: ASR Adapters, Normalization, And Consensus

- [ ] Implement adapter interface with `preflight()` and `transcribe_batch()`.
- [ ] Implement Whisper adapter using `faster_whisper` when available.
- [ ] Implement Qwen adapter using `qwen_asr.Qwen3ASRModel` when available.
- [ ] Implement Cohere and Granite adapters using local `transformers` classes when available.
- [ ] Preflight must fail for enabled missing model paths or imports.
- [ ] Add fixture ASR adapter for tests.
- [ ] Normalize every model transcript.
- [ ] Strong consensus requires 3 of 4 successful models by default.
- [ ] Weak consensus selects best available transcript by configurable model priority and quality score.
- [ ] Add tests for normalization, strong consensus, weak consensus, and model failure isolation.

## Task 6: PII, Alignment, Mask Plans, Redaction, Validation

- [ ] Implement regex/rules baseline for email, phone, SSN, DOB, ZIP, account/policy/member IDs, and name cue patterns.
- [ ] Add optional GLiNER, Piiranha, and spaCy detectors with strict dependency checks only when enabled.
- [ ] Run PII on final/best-effort transcript by default; support all-model-transcript mode.
- [ ] Implement required aligner interface. Test aligner maps transcript tokens to segment time spans deterministically.
- [ ] Fail PII-relevant segments/calls when alignment fails.
- [ ] Merge mask intervals with 80 ms pre-padding and 120 ms post-padding.
- [ ] Apply beep/silence/noise masks to only planned channel intervals.
- [ ] Export original format via FFmpeg when possible, WAV fallback otherwise, and record `completed_with_fallback`.
- [ ] Validate duration, channel count, readability, and fallback status.
- [ ] Add tests for PII detection, alignment failure, channel-only masking, duration preservation, and WAV fallback metadata.

## Task 7: Manifests, Reports, CLI, And Orchestration

- [ ] Implement global and per-year CSV/JSONL manifests with all required segment fields.
- [ ] Add optional Parquet/Excel export if pandas/openpyxl/pyarrow are installed.
- [ ] Implement `dry-run`, `run`, `resume`, `run-stage`, `retry-failed`, `validate`, `audit`, `pause`, and `clean-cache`.
- [ ] Implement stage dependency checks so stage-by-stage runs fail early with actionable messages.
- [ ] Implement resource report with CPU/RAM/GPU/disk information.
- [ ] Add CLI smoke tests for dry-run, run-stage preprocess, pause, and manifest generation.

## Task 8: Verification

- [ ] Run `python3 -m compileall amc_pipeline tests`.
- [ ] Run `python3 -m unittest discover -s tests -v`.
- [ ] Run CLI smoke commands against generated tiny fixtures.
- [ ] Inspect outputs for mirrored paths, segment paths, manifest rows, state rows, and validation reports.
- [ ] Record any missing external runtime dependency as preflight behavior, not a silent success.

## Self-Review Notes

- The plan covers the approved spec while keeping the package runnable in the current bare workspace.
- The implementation must not import heavy ML libraries at module import time.
- The implementation must never overwrite the input folder.
- The implementation must treat original-format export as FFmpeg-dependent and must produce WAV fallback when original export is unavailable.
- The implementation must preserve experiment-derived model paths in config defaults.
