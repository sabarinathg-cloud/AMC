from __future__ import annotations

import json
import os
import re
import shutil
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .alignment import RequiredAlignmentError, TokenUniformAligner, TorchaudioCTCAligner, WhisperXAligner
from .audio import MissingDependencyError, decode_to_wav, encode_from_wav, temp_wav_path
from .config import PipelineConfig
from .consensus import DEFAULT_AGREEMENT_MODELS, DEFAULT_PRIORITY, UNREADABLE_METHODS, build_consensus, partition_degenerate
from .discovery import discover_audio_files
from .inspection import inspect_audio
from .manifests import write_segment_manifests
from .masking import redact_wav_with_plan
from .models import ASRResult, AudioFileRecord, MaskInterval, PIISpan, SegmentRecord, dataclass_to_dict
from .normalization import normalize_transcript
from .pii_detection import build_enabled_detectors, is_maskable_entity, preflight_detectors
from .preprocessing import preprocess_file
from .progress import iter_progress
from .resources import detect_resources
from .segmentation import preflight_vad_backend
from .speaker import SpeakerParams, embed_call_channels, load_cluster_map, write_shard_embeddings
from .state import PostgresStateStore, SQLiteStateStore
from .transcription import LANGUAGE_DETECTING_MODELS, build_enabled_adapters, language_source_model
from .validation import validate_audio_pair


# How many empty alignments the align stage buffers before writing them. Segments with
# no PII spans need nothing aligned, just an artifact row saying so, and they are the
# large majority -- so their two autocommit transactions each (artifact + failure clear)
# were most of the stage's wall time. Segments that do get aligned are still written one
# at a time: the GPU work dwarfs the write, and buffering them would only widen the
# window of expensive work to redo after a crash.
_ALIGN_EMPTY_FLUSH_ROWS = 1000

# How many normalized transcripts the normalize stage buffers before writing them
# back. Each flush is one transaction on the Lustre-backed WAL DB, and per-row
# autocommit was the whole cost of that stage. The stage is idempotent (it skips
# rows that already carry `normalized_transcript`), so a crash mid-buffer only
# redoes normalization for at most this many rows.
_NORMALIZE_FLUSH_ROWS = 1000


def _chunk_segments_by_call(segments: list[SegmentRecord], target: int):
    """Yield lists of segments grouped so whole calls stay together.

    Segments are bucketed by call (falling back to file, then segment id),
    preserving first-seen order, then whole calls are accumulated into a chunk
    until it reaches ``target`` segments. Keeping calls intact means Whisper's
    full-call batched path still receives complete calls, while the ASR stage
    can persist/resume at chunk granularity. A single call larger than
    ``target`` is emitted as its own chunk rather than being split.
    """
    buckets: dict[str, list[SegmentRecord]] = {}
    order: list[str] = []
    for s in segments:
        key = getattr(s, "call_id", None) or getattr(s, "file_id", None) or s.segment_id
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(s)
    chunk: list[SegmentRecord] = []
    for key in order:
        chunk.extend(buckets[key])
        if len(chunk) >= target:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _failed_results(model_name: str, segments: list[SegmentRecord]) -> list[ASRResult]:
    """Build placeholder error results so a failed chunk persists as ``failed``.

    These rows are written with status ``failed`` and are therefore retried on
    the next run (the resume filter only keeps ``transcribed`` rows)."""
    return [
        ASRResult(
            segment_id=s.segment_id,
            model_name=model_name,
            transcript="",
            confidence=0.0,
            language=s.language,
            language_confidence=s.language_confidence,
            error="model_stage_failure",
            raw={"model_stage_failure": True},
        )
        for s in segments
    ]


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.ensure_output_dirs()
        if config.state.backend == "sqlite":
            self.state = SQLiteStateStore(config.state_db_path)
        elif config.state.backend == "postgres":
            self.state = PostgresStateStore(config.state.postgres_dsn or "")
        else:
            raise RuntimeError(f"Unsupported state backend: {config.state.backend}")

    def dry_run(self) -> dict[str, Any]:
        records = discover_audio_files(self.config)
        inspected = []
        errors = []
        for record in iter_progress(records, desc="Inspect audio", total=len(records), unit="file", enabled=self.config.progress_enabled):
            try:
                meta = inspect_audio(record.source_path)
                inspected.append({"file_id": record.file_id, "channels": meta.channels, "duration_sec": meta.duration_sec, "sample_rate": meta.sample_rate})
                payload = dataclass_to_dict(record)
                payload["audio_metadata"] = _compact_audio_metadata(meta)
                self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "inspected", "payload": payload})
            except Exception as exc:
                errors.append({"file_id": record.file_id, "source_path": str(record.source_path), "error": repr(exc)})
                self.state.record_failure(f"{record.file_id}:inspect", "file", record.file_id, repr(exc), retryable=True)
        summary = {
            "total_files": len(records),
            "inspectable_files": len(inspected),
            "errors": errors,
            "preflight": self.preflight_status(),
            "resources": detect_resources(self.config.output_root),
            "config_hash": self.config.config_hash(),
        }
        self._write_report("dry_run_summary.json", summary)
        return summary

    def preflight_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {"preprocessing": {}, "asr": {}, "pii": {}, "alignment": {}}
        try:
            preflight_vad_backend(self.config.audio.vad_backend, self.config.audio.silero_repo_or_dir)
            status["preprocessing"][self.config.audio.vad_backend] = {"ok": True}
        except Exception as exc:
            status["preprocessing"][self.config.audio.vad_backend] = {"ok": False, "error": str(exc)}
        for adapter in build_enabled_adapters(self.config):
            try:
                adapter.preflight()
                status["asr"][adapter.name] = {"ok": True}
            except Exception as exc:
                status["asr"][adapter.name] = {"ok": False, "error": str(exc)}
        for detector in build_enabled_detectors(self.config):
            try:
                detector.preflight()
                status["pii"][detector.name] = {"ok": True}
            except Exception as exc:
                status["pii"][detector.name] = {"ok": False, "error": str(exc)}
        try:
            if self.config.alignment.backend == "whisperx":
                WhisperXAligner().preflight()
            elif self.config.alignment.backend == "torchaudio_ctc":
                TorchaudioCTCAligner().preflight()
            status["alignment"][self.config.alignment.backend] = {"ok": True}
        except Exception as exc:
            status["alignment"][self.config.alignment.backend] = {"ok": False, "error": str(exc)}
        return status

    def run_all(self) -> dict[str, Any]:
        self.raise_for_failed_preflight()
        summaries: dict[str, Any] = {}
        stages = ["preprocess", "asr", "normalize", "consensus", "pii", "align", "mask-plan", "redact", "validate", "manifest"]
        for stage in iter_progress(stages, desc="Pipeline stages", total=len(stages), unit="stage", enabled=self.config.progress_enabled, leave=True):
            summaries[stage] = self.run_stage(stage)
        self._write_report("run_summary.json", summaries)
        return summaries

    def raise_for_failed_preflight(self) -> None:
        status = self.preflight_status()
        failures = []
        for group, checks in status.items():
            for name, check in checks.items():
                if not check.get("ok"):
                    failures.append(f"{group}.{name}: {check.get('error')}")
        if failures:
            raise RuntimeError("Preflight failed before processing: " + "; ".join(failures))

    def run_stage(self, stage: str) -> dict[str, Any]:
        if stage in {"discover", "inspect", "dry-run"}:
            return self.dry_run()
        if stage == "preprocess":
            return self.preprocess()
        if stage == "asr":
            return self.asr()
        if stage == "normalize":
            return self.normalize()
        if stage == "consensus":
            return self.consensus()
        if stage in {"agreement", "model-agreement", "model_agreement"}:
            return self.model_agreement()
        if stage in {"speaker-embed", "speaker_embed"}:
            return self.speaker_embed()
        if stage in {"speaker-assign", "speaker_assign"}:
            return self.speaker_assign()
        if stage == "pii":
            return self.pii()
        if stage == "align":
            return self.align()
        if stage in {"mask-plan", "mask_plan"}:
            return self.mask_plan()
        if stage in {"redact", "mask"}:
            return self.redact()
        if stage == "validate":
            return self.validate()
        if stage == "manifest":
            return self.write_manifests()
        raise ValueError(f"Unsupported stage: {stage}")

    def preprocess(self) -> dict[str, Any]:
        records = discover_audio_files(self.config)
        all_segments = 0
        errors: list[dict[str, Any]] = []
        skipped = 0
        existing_files: dict[str, dict[str, Any]] = {row["file_id"]: row for row in self.state.iter_files()}
        existing_segment_file_ids: set[str] = {row["file_id"] for row in self.state.iter_segments()}
        # Reconcile stale failures: a file that is now `preprocessed` but still carries a
        # failure row (e.g. recorded during an OOM attempt, then succeeded on a later retry)
        # should not keep inflating the `failures` count. Clear those up front so a resume
        # also cleans the backlog from before this behavior existed.
        resolved_failures: list[str] = []
        for file_id, row in existing_files.items():
            if row.get("status") == "preprocessed" and file_id in existing_segment_file_ids:
                resolved_failures.append(f"{file_id}:preprocess")
                resolved_failures.append(f"{file_id}:inspect")
        self.state.clear_failures(resolved_failures)
        pending: list[AudioFileRecord] = []
        for record in records:
            if existing_files.get(record.file_id, {}).get("status") == "preprocessed" and record.file_id in existing_segment_file_ids:
                skipped += 1
                continue
            pending.append(record)
        workers = self.config.resolved_workers(self.config.preprocess_workers)

        def file_payload(record: AudioFileRecord) -> dict[str, Any]:
            payload = dataclass_to_dict(record)
            prior = existing_files.get(record.file_id, {}).get("payload") or {}
            meta = prior.get("audio_metadata")
            if meta is None:
                try:
                    meta = _compact_audio_metadata(inspect_audio(record.source_path))
                except Exception:
                    meta = None
            if meta is not None:
                payload["audio_metadata"] = meta
            return payload

        def persist(record: AudioFileRecord, segments) -> int:
            self.state.upsert_segments_many(
                [(segment.segment_id, record.file_id, "preprocessed", dataclass_to_dict(segment)) for segment in segments]
            )
            self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "preprocessed", "payload": file_payload(record)})
            self.state.clear_failures([f"{record.file_id}:preprocess", f"{record.file_id}:inspect"])
            return len(segments)

        if workers <= 1 or len(pending) <= 1:
            for record in iter_progress(pending, desc="Preprocess audio", total=len(pending), unit="file", enabled=self.config.progress_enabled):
                if self.state.should_pause(self.config.run_id):
                    break
                try:
                    self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "preprocessing", "payload": file_payload(record)})
                    all_segments += persist(record, preprocess_file(record, self.config))
                except Exception as exc:
                    errors.append({"file_id": record.file_id, "source_path": str(record.source_path), "error": repr(exc)})
                    self.state.record_failure(f"{record.file_id}:preprocess", "file", record.file_id, repr(exc), retryable=True)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Do the full per-file cost -- ffprobe metadata + decode + GPU VAD --
            # inside the worker so the main thread never serializes on ffprobe (the
            # old submit loop did inspect_audio per file on the main thread, which
            # capped the whole stage at ~1 file/s). The main thread only does the
            # SQLite writes (single-writer) as results arrive. We keep a bounded
            # in-flight window so results are persisted incrementally -- segments
            # land in the DB as files finish (live progress + true resumability),
            # not only after all ~8k tasks are submitted -- and memory stays flat.
            def job(record: AudioFileRecord):
                segments = preprocess_file(record, self.config)
                return record, segments, file_payload(record)

            max_inflight = max(workers * 2, workers + 2)
            pending_iter = iter(pending)
            inflight: dict[Any, AudioFileRecord] = {}
            paused = False

            def submit_next(pool) -> None:
                nonlocal paused
                if paused:
                    return
                record = next(pending_iter, None)
                if record is None:
                    return
                inflight[pool.submit(job, record)] = record

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in range(max_inflight):
                    submit_next(pool)
                progress = iter_progress(
                    range(len(pending)), desc="Preprocess audio", total=len(pending), unit="file", enabled=self.config.progress_enabled
                )
                progress_it = iter(progress)
                while inflight:
                    future = next(as_completed(list(inflight)))
                    record = inflight.pop(future)
                    try:
                        rec_done, segments, payload = future.result()
                        self.state.upsert_segments_many(
                            [(segment.segment_id, rec_done.file_id, "preprocessed", dataclass_to_dict(segment)) for segment in segments]
                        )
                        self.state.upsert_file({"file_id": rec_done.file_id, "source_path": str(rec_done.source_path), "status": "preprocessed", "payload": payload})
                        self.state.clear_failures([f"{rec_done.file_id}:preprocess", f"{rec_done.file_id}:inspect"])
                        all_segments += len(segments)
                    except Exception as exc:
                        errors.append({"file_id": record.file_id, "source_path": str(record.source_path), "error": repr(exc)})
                        self.state.record_failure(f"{record.file_id}:preprocess", "file", record.file_id, repr(exc), retryable=True)
                    try:
                        next(progress_it)
                    except StopIteration:
                        pass
                    if not paused and self.state.should_pause(self.config.run_id):
                        paused = True
                    submit_next(pool)
        segment_count = self.state.count_segments()
        manifest_paths = (
            write_segment_manifests(
                self.config.output_root,
                self._iter_segment_records(),
                enable_dataframe_exports=self.config.manifest_dataframe_exports,
                xlsx_max_rows=self.config.manifest_xlsx_max_rows,
                write_csv=self.config.manifest_write_csv,
                write_per_year=self.config.manifest_write_per_year,
                parquet_batch_size=self.config.manifest_parquet_batch_size,
            )
            if segment_count and not self.config.skip_stage_manifest
            else []
        )
        summary = {"stage": "preprocess", "files": len(records), "segments": segment_count, "new_segments": all_segments, "skipped_files": skipped, "errors": errors, "manifests": [str(p) for p in manifest_paths]}
        self._write_report("preprocess_summary.json", summary)
        return summary

    def asr(self) -> dict[str, Any]:
        segments = self._load_segments()
        adapters = build_enabled_adapters(self.config)
        counts: dict[str, int] = {}
        skipped_counts: dict[str, int] = {}
        failed_models: list[dict[str, Any]] = []
        supported = {x.lower() for x in self.config.languages}
        language_by_segment: dict[str, str | None] = {}
        language_confidence_by_segment: dict[str, float | None] = {}
        segment_by_id = {s.segment_id: s for s in segments}
        existing_results = {
            (row["segment_id"], row["model_name"]): row
            for row in self.state.fetch_model_results()
            if row.get("status") == "transcribed"
        }
        language_source = language_source_model(
            (adapter.name for adapter in adapters),
            (model_name for (_, model_name) in existing_results),
        )
        for (segment_id, model_name), row in existing_results.items():
            if model_name != language_source:
                continue
            payload = row["payload"]
            language_by_segment[segment_id] = payload.get("language")
            language_confidence_by_segment[segment_id] = payload.get("language_confidence")
        adapters = sorted(adapters, key=lambda a: 0 if a.name == language_source else 1)
        for adapter in adapters:
            adapter.progress_enabled = self.config.progress_enabled
        # Persist transcripts in call-grouped chunks instead of one bulk write at
        # the end of the whole pass. This makes the ASR stage:
        #   * resumable mid-pass -- a crash/interruption N hours in keeps every
        #     chunk already written (the start-of-stage `existing_results` filter
        #     skips them on restart), instead of redoing the entire model; and
        #   * observable live -- `model_results` climbs as chunks land, like the
        #     preprocess `segments` count.
        # Chunks respect call boundaries so Whisper's full-call batched path still
        # sees whole calls. Tune the chunk size with AMC_ASR_PERSIST_CHUNK.
        chunk_target = max(1, int(os.environ.get("AMC_ASR_PERSIST_CHUNK", "256") or "256"))

        def _persist_chunk(model_name: str, results: list[ASRResult]) -> None:
            seg_rows: list[tuple[str, str, str, dict[str, Any]]] = []
            model_rows: list[tuple[str, str, str, dict[str, Any]]] = []
            for result in results:
                if model_name == language_source:
                    language_by_segment[result.segment_id] = result.language
                    language_confidence_by_segment[result.segment_id] = result.language_confidence
                    segment = segment_by_id.get(result.segment_id)
                    if segment is not None:
                        payload = dataclass_to_dict(replace(segment, language=result.language, language_confidence=result.language_confidence))
                        seg_rows.append((segment.segment_id, segment.file_id, segment.status, payload))
                model_rows.append(
                    (result.segment_id, result.model_name, "failed" if result.error else "transcribed", dataclass_to_dict(result))
                )
            if seg_rows:
                self.state.upsert_segments_many(seg_rows)
            self.state.upsert_model_results_many(model_rows)

        for adapter in iter_progress(adapters, desc="ASR models", total=len(adapters), unit="model", enabled=self.config.progress_enabled):
            try:
                model_segments = segments
                if adapter.name != language_source and language_by_segment:
                    model_segments = [
                        replace(
                            s,
                            language=language_by_segment.get(s.segment_id),
                            language_confidence=language_confidence_by_segment.get(s.segment_id),
                        )
                        for s in segments
                        if (language_by_segment.get(s.segment_id) or "").lower() in supported
                    ]
                skipped_counts[adapter.name] = sum(1 for s in model_segments if (s.segment_id, adapter.name) in existing_results)
                model_segments = [s for s in model_segments if (s.segment_id, adapter.name) not in existing_results]
                if not model_segments:
                    counts[adapter.name] = 0
                    continue
                try:
                    adapter.preflight()
                    preflight_ok = True
                except Exception as exc:
                    tb = traceback.format_exc()
                    failed_models.append({"model": adapter.name, "error": repr(exc), "segments": len(model_segments)})
                    self.state.record_failure(f"asr:{adapter.name}", "model", adapter.name, repr(exc), retryable=True, traceback=tb)
                    preflight_ok = False

                produced = 0
                for chunk in _chunk_segments_by_call(model_segments, chunk_target):
                    if preflight_ok:
                        try:
                            results = adapter.transcribe_batch(chunk)
                            self.state.clear_failures([f"asr:{adapter.name}", f"asr:{adapter.name}:chunk"])
                        except Exception as exc:
                            tb = traceback.format_exc()
                            failed_models.append({"model": adapter.name, "error": repr(exc), "segments": len(chunk)})
                            self.state.record_failure(f"asr:{adapter.name}:chunk", "model", adapter.name, repr(exc), retryable=True, traceback=tb)
                            results = _failed_results(adapter.name, chunk)
                    else:
                        results = _failed_results(adapter.name, chunk)
                    # Persist this chunk before starting the next one: on restart the
                    # start-of-stage existing_results filter skips these segments.
                    _persist_chunk(adapter.name, results)
                    produced += len(results)
                counts[adapter.name] = produced
            finally:
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()
        skipped = []
        for segment in segments:
            lang = language_by_segment.get(segment.segment_id)
            if lang and lang.lower() not in supported:
                skipped.append({"segment_id": segment.segment_id, "language": lang, "reason": "unsupported_language"})
                payload = dataclass_to_dict(segment)
                payload["language"] = lang
                payload["status_reason"] = "unsupported_language"
                self.state.upsert_segment(segment.segment_id, segment.file_id, "skipped_unsupported_language", payload)
        summary = {"stage": "asr", "segments": len(segments), "models": counts, "skipped_existing": skipped_counts, "model_failures": failed_models, "skipped_unsupported_language": skipped}
        self._write_report("asr_summary.json", summary)
        return summary

    def normalize(self) -> dict[str, Any]:
        total = self.state.count_model_results()
        count = 0
        skipped = 0
        pending: list[tuple[str, str, str, dict[str, Any]]] = []
        for row in iter_progress(self.state.iter_model_results(), desc="Normalize transcripts", total=total, unit="result", enabled=self.config.progress_enabled):
            payload = row["payload"]
            if "normalized_transcript" in payload:
                skipped += 1
                continue
            payload["normalized_transcript"] = normalize_transcript(payload.get("transcript", ""), remove_fillers=True)
            pending.append((row["segment_id"], row["model_name"], row["status"], payload))
            count += 1
            if len(pending) >= _NORMALIZE_FLUSH_ROWS:
                self.state.upsert_model_results_many(pending)
                pending = []
        if pending:
            self.state.upsert_model_results_many(pending)
        summary = {"stage": "normalize", "model_results": count, "skipped_existing": skipped}
        self._write_report("normalize_summary.json", summary)
        return summary

    def consensus(self) -> dict[str, Any]:
        grouped: dict[str, list[ASRResult]] = {}
        for row in self.state.iter_model_results():
            payload = row["payload"]
            grouped.setdefault(row["segment_id"], []).append(
                ASRResult(
                    segment_id=row["segment_id"],
                    model_name=row["model_name"],
                    transcript=payload.get("transcript", ""),
                    confidence=float(payload.get("confidence") or 0.0),
                    language=payload.get("language"),
                    language_confidence=payload.get("language_confidence"),
                    error=payload.get("error"),
                    raw=payload.get("raw", {}),
                )
            )
        strong = 0
        skipped = 0
        existing = {
            artifact["payload"].get("segment_id") or artifact["artifact_id"].split(":", 1)[-1]
            for artifact in self.state.iter_artifacts("consensus")
            if artifact.get("status") == "completed"
        }
        # Duration turns a transcript's length into a speech rate, which is how a
        # looping model is told apart from a talkative one (see `degeneracy`).
        duration_by_segment = {s.segment_id: s.duration_sec for s in self._load_segments()}
        rejected_by_model: Counter = Counter()
        rejected_rows: list[dict[str, Any]] = []
        items = list(grouped.items())
        for segment_id, results in iter_progress(items, desc="Build consensus", total=len(items), unit="segment", enabled=self.config.progress_enabled):
            if segment_id in existing:
                skipped += 1
                continue
            duration_sec = duration_by_segment.get(segment_id)
            _, degenerate = partition_degenerate(results, duration_sec)
            for result, report in degenerate:
                rejected_by_model[result.model_name] += 1
                rejected_rows.append(
                    {
                        "segment_id": segment_id,
                        "model": result.model_name,
                        "reason": report.reason,
                        "chars": len(result.transcript or ""),
                        "chars_per_sec": round(report.chars_per_sec, 1) if report.chars_per_sec else None,
                        "repeat_ratio": round(report.repeat_ratio, 3),
                    }
                )
            result = build_consensus(results, min_successful_models=self.config.consensus_min_models, duration_sec=duration_sec)
            if result.strong:
                strong += 1
            self.state.record_artifact(f"consensus:{segment_id}", "consensus", Path(segment_id), "completed", dataclass_to_dict(result))
        if rejected_rows:
            self._write_report("degenerate_asr.json", rejected_rows)
        processed = len(grouped) - skipped
        summary = {"stage": "consensus", "segments": len(grouped), "processed": processed, "skipped_existing": skipped, "strong": strong, "weak": processed - strong, "degenerate_rejected": dict(rejected_by_model)}
        self._write_report("consensus_summary.json", summary)
        return summary

    def model_agreement(self) -> dict[str, Any]:
        extras = self._manifest_extras()
        buckets = {"all_4_agree": 0, "three_agree": 0, "two_agree": 0, "all_different": 0}
        present_hist: Counter = Counter()
        four_model = 0
        for ex in extras.values():
            mp = ex.get("models_present")
            if isinstance(mp, int):
                present_hist[mp] += 1
            label = ex.get("model_agreement")
            if label in buckets:
                four_model += 1
                buckets[label] += 1
        paths = write_segment_manifests(
            self.config.output_root,
            self._iter_segment_records(),
            extras,
            enable_dataframe_exports=self.config.manifest_dataframe_exports,
            xlsx_max_rows=self.config.manifest_xlsx_max_rows,
            write_csv=self.config.manifest_write_csv,
            write_per_year=self.config.manifest_write_per_year,
            parquet_batch_size=self.config.manifest_parquet_batch_size,
        )
        summary = {
            "stage": "model_agreement",
            "models": self._agreement_panel(),
            "segments_total": len(extras),
            "segments_with_4_models": four_model,
            "agreement_4model": buckets,
            "usable_models_per_segment": {str(k): present_hist[k] for k in sorted(present_hist)},
            "match": "exact equality of normalized_transcript; bucket = largest identical cluster",
            "manifest_paths": [str(p) for p in paths],
        }
        self._write_report("model_agreement_summary.json", summary)
        return summary

    def speaker_embed(self) -> dict[str, Any]:
        """Per-shard GPU stage: one robust ReDimNet centroid per (call_id, channel).

        Writes ``<speaker_embed_root>/shard-<shard_index>.npz`` on shared storage. The
        run-once ``cluster-speakers`` step then gathers every shard's centroids to assign
        globally-consistent speaker ids. Re-running regenerates the shard file from the DB.
        """
        embed_root = self.config.speaker_embed_root
        if embed_root is None:
            raise RuntimeError("speaker_embed requires --speaker-embed-root (a shared directory).")
        shard_index = self.config.discovery.shard_index
        shard_index = 0 if shard_index is None else int(shard_index)
        params = SpeakerParams.from_config(self.config)
        data = embed_call_channels(
            self._iter_segment_records(),
            params,
            progress_enabled=self.config.progress_enabled,
        )
        out_path = Path(embed_root) / f"shard-{shard_index}.npz"
        write_shard_embeddings(out_path, data)
        summary = {
            "stage": "speaker_embed",
            "shard_index": shard_index,
            "call_sides": int(len(data["call_id"])),
            "embedding_dim": int(data["centroids"].shape[1]) if data["centroids"].size else 0,
            "max_segments_per_side": params.max_segments_per_side,
            "model": f"ReDimNet-{params.model_name}/{params.train_type}/{params.dataset}",
            "output": str(out_path),
        }
        self._write_report("speaker_embed_summary.json", summary)
        return summary

    def speaker_assign(self) -> dict[str, Any]:
        """Per-shard stage: join the global speaker_cluster_id onto this shard's segments.

        Reads the global mapping written by ``cluster-speakers`` and regenerates the manifest
        so ``all_segments.parquet`` gains a ``speaker_cluster_id`` column. The join happens in
        ``_manifest_extras`` (which loads ``clusters.parquet`` directly); sides missing from the
        global map (e.g. no usable audio) get a deterministic fallback id so every segment is
        labeled. We intentionally do NOT persist a per-side artifact into SQLite here -- doing so
        was ~1 insert per (call_id, channel) into FSx-backed SQLite (a write storm of hundreds of
        thousands of rows per shard) that stalled workers for over an hour.
        """
        clusters_path = self.config.speaker_clusters_path
        if clusters_path is None:
            raise RuntimeError("speaker_assign requires --speaker-clusters (the global clusters.parquet).")
        cluster_map = load_cluster_map(Path(clusters_path))
        assigned = 0
        fallback = 0
        seen: set[tuple[str, str]] = set()
        for row in self.state.iter_segments():
            payload = row.get("payload") or {}
            key = (str(payload.get("call_id", "")), str(int(payload.get("channel", 0))))
            if key in seen:
                continue
            seen.add(key)
            if cluster_map.get(key) is not None:
                assigned += 1
            else:
                fallback += 1
        paths = write_segment_manifests(
            self.config.output_root,
            self._iter_segment_records(),
            self._manifest_extras(),
            enable_dataframe_exports=self.config.manifest_dataframe_exports,
            xlsx_max_rows=self.config.manifest_xlsx_max_rows,
            write_csv=self.config.manifest_write_csv,
            write_per_year=self.config.manifest_write_per_year,
            parquet_batch_size=self.config.manifest_parquet_batch_size,
        )
        summary = {
            "stage": "speaker_assign",
            "clusters_path": str(clusters_path),
            "call_sides_assigned": assigned,
            "call_sides_fallback": fallback,
            "manifest_paths": [str(p) for p in paths],
        }
        self._write_report("speaker_assign_summary.json", summary)
        return summary

    def pii(self) -> dict[str, Any]:
        detectors = build_enabled_detectors(self.config)
        preflight_detectors(detectors)
        total = self.state.count_artifacts("consensus")
        existing = {
            artifact["payload"].get("segment_id") or artifact["artifact_id"].split(":", 1)[-1]
            for artifact in self.state.iter_artifacts("pii")
            if artifact.get("status") == "completed"
        }
        count = 0
        skipped = 0
        # Buffer segments and run each detector once over the whole batch so the neural
        # NER models (GLiNER/Piiranha on GPU, spaCy via nlp.pipe) saturate the device
        # instead of doing one tiny forward pass per segment. Output is byte-identical to
        # the per-segment path: results are scattered back in the same candidate/detector
        # order before dedupe.
        batch_size = max(1, int(os.environ.get("AMC_PII_SEGMENT_BATCH", "128")))
        buffer: list[tuple[str, str]] = []

        def _flush() -> int:
            if not buffer:
                return 0
            n = 0
            for (segment_id, _), span_dicts in zip(buffer, self._detect_pii_for_batch(buffer, detectors)):
                self.state.record_artifact(f"pii:{segment_id}", "pii", Path(segment_id), "completed", {"segment_id": segment_id, "spans": span_dicts})
                n += len(span_dicts)
            buffer.clear()
            return n

        for artifact in iter_progress(self.state.iter_artifacts("consensus"), desc="Detect PII", total=total, unit="segment", enabled=self.config.progress_enabled):
            payload = artifact["payload"]
            final_transcript = payload.get("final_transcript", "")
            segment_id = payload.get("segment_id") or artifact["artifact_id"].split(":", 1)[-1]
            if segment_id in existing:
                skipped += 1
                continue
            buffer.append((segment_id, final_transcript))
            if len(buffer) >= batch_size:
                count += _flush()
        count += _flush()
        summary = {"stage": "pii", "segments": total, "processed": total - skipped, "skipped_existing": skipped, "spans": count}
        self._write_report("pii_summary.json", summary)
        return summary

    def _candidates_for_segment(self, segment_id: str, final_transcript: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = [("final", final_transcript)]
        if self.config.pii_on_all_model_transcripts:
            for row in self.state.fetch_model_results(segment_id):
                text = row["payload"].get("transcript", "")
                if text:
                    candidates.append((row["model_name"], text))
        return candidates

    def _detect_pii_for_batch(self, items: list[tuple[str, str]], detectors) -> list[list[dict[str, Any]]]:
        # Build a flat list of every candidate transcript across the batch, remembering which
        # (segment, source) each one belongs to, so we can run each detector once over all of them.
        flat_texts: list[str] = []
        per_item: list[list[tuple[str, str, int]]] = []  # [(source_name, transcript, flat_index), ...] per item
        for segment_id, final_transcript in items:
            refs: list[tuple[str, str, int]] = []
            for source_name, transcript in self._candidates_for_segment(segment_id, final_transcript):
                refs.append((source_name, transcript, len(flat_texts)))
                flat_texts.append(transcript)
            per_item.append(refs)

        # One batched forward pass per detector over the whole flat list.
        detector_spans: list[list[list[PIISpan]]] = [detector.detect_batch(flat_texts) for detector in detectors]

        out: list[list[dict[str, Any]]] = []
        for (segment_id, final_transcript), refs in zip(items, per_item):
            span_dicts: list[dict[str, Any]] = []
            for source_name, transcript, flat_idx in refs:
                for det_idx, detector in enumerate(detectors):
                    for span in detector_spans[det_idx][flat_idx]:
                        # Policy: skip non-identifying clinical entities (condition/medication/
                        # lab result). Identifying PII -- incl. re-identifiable medical codes --
                        # still masked. Override with AMC_MASK_MEDICAL_DETAILS=1.
                        if not is_maskable_entity(span.entity_type):
                            continue
                        mapped = _map_span_to_final(span, transcript, final_transcript)
                        if mapped is None:
                            continue
                        mapped = _expand_numeric_pii_context(mapped, final_transcript)
                        data = dataclass_to_dict(mapped)
                        data["transcript_source"] = source_name
                        data["detector_source"] = span.source
                        span_dicts.append(data)
            out.append(_dedupe_pii_dicts(span_dicts))
        return out

    def _detect_pii_for_segment(self, segment_id: str, final_transcript: str, detectors) -> list[dict[str, Any]]:
        # Single-segment path (kept for callers/tests); delegates to the batched implementation.
        return self._detect_pii_for_batch([(segment_id, final_transcript)], detectors)[0]

    def align(self) -> dict[str, Any]:
        if self.config.alignment.backend == "torchaudio_ctc":
            TorchaudioCTCAligner().preflight()
            raise RuntimeError("torchaudio_ctc preflight passed, but runtime alignment model wiring must be configured for the deployment")
        elif self.config.alignment.backend == "whisperx":
            aligner = WhisperXAligner()
            aligner.preflight()
        elif self.config.alignment.backend == "token_uniform":
            aligner = TokenUniformAligner()
        else:
            raise RuntimeError(f"Unsupported alignment backend: {self.config.alignment.backend}")
        consensus_by_segment = {a["payload"]["segment_id"]: a["payload"] for a in self.state.fetch_artifacts("consensus")}
        segment_by_id = {s.segment_id: s for s in self._load_segments()}
        existing = {
            artifact["payload"].get("segment_id") or artifact["artifact_id"].split(":", 1)[-1]
            for artifact in self.state.fetch_artifacts("alignment")
            if artifact.get("status") == "completed"
        }
        aligned = 0
        failures = 0
        skipped = 0
        degraded = 0
        unreadable = 0
        # Supported language roots (e.g. {"en", "es"}). Whisper occasionally misdetects
        # short/noisy segments as other languages (pt, nn, ...). Forced alignment for those
        # would fetch a language-specific wav2vec2 model from HuggingFace, which (a) may not
        # exist / lack safetensors and (b) is pointless for an out-of-scope language.
        supported_langs = {str(x).lower().replace("_", "-").split("-", 1)[0] for x in self.config.languages}
        # Uniform fallback needs no language model and still maps PII spans to time
        # intervals, so masking stays fail-safe (no leak) even when a forced-alignment
        # model is unavailable or the detected language is unsupported.
        uniform_fallback = TokenUniformAligner()
        # Buffered writes for the no-PII-span case; see _ALIGN_EMPTY_FLUSH_ROWS.
        empty_artifacts: list[tuple[str, str, Path, str, dict[str, Any] | None]] = []
        empty_failure_ids: list[str] = []

        def flush_empty_alignments() -> None:
            if not empty_artifacts:
                return
            self.state.record_artifacts_many(empty_artifacts)
            self.state.clear_failures(empty_failure_ids)
            empty_artifacts.clear()
            empty_failure_ids.clear()

        pii_artifacts = self.state.fetch_artifacts("pii")
        for artifact in iter_progress(pii_artifacts, desc="Force alignment", total=len(pii_artifacts), unit="segment", enabled=self.config.progress_enabled):
            payload = artifact["payload"]
            segment_id = payload["segment_id"]
            if segment_id in existing:
                skipped += 1
                continue
            segment = segment_by_id.get(segment_id)
            consensus = consensus_by_segment.get(segment_id)
            if segment is None or consensus is None:
                continue
            spans = [_pii_span_from_payload(span) for span in payload.get("spans", [])]
            if str(consensus.get("consensus_method") or "") in UNREADABLE_METHODS:
                # No model produced a usable transcript for this segment (all errored,
                # or all looped -- see `degeneracy`), so there is no text to search for
                # PII and an empty span list here means "nothing to mask". Mask the whole
                # segment instead: losing one segment of audio beats shipping speech that
                # nothing has read.
                interval = MaskInterval(segment.channel, 0.0, float(segment.duration_sec), "no_usable_transcript", "UNKNOWN", 1.0, "fail_safe")
                empty_artifacts.append(
                    (f"alignment:{segment_id}", "alignment", Path(segment_id), "completed",
                     {"segment_id": segment_id, "words": [], "intervals": [dataclass_to_dict(interval)]})
                )
                empty_failure_ids.append(f"{segment_id}:alignment")
                unreadable += 1
                if len(empty_artifacts) >= _ALIGN_EMPTY_FLUSH_ROWS:
                    flush_empty_alignments()
                continue
            if not spans:
                empty_artifacts.append(
                    (f"alignment:{segment_id}", "alignment", Path(segment_id), "completed", {"segment_id": segment_id, "words": [], "intervals": []})
                )
                empty_failure_ids.append(f"{segment_id}:alignment")
                if len(empty_artifacts) >= _ALIGN_EMPTY_FLUSH_ROWS:
                    flush_empty_alignments()
                continue
            try:
                transcript = consensus.get("final_transcript", "")
                if isinstance(aligner, WhisperXAligner):
                    seg_lang = segment.language or consensus.get("language") or "en"
                    lang_root = str(seg_lang).lower().replace("_", "-").split("-", 1)[0]
                    use_uniform = lang_root not in supported_langs
                    if not use_uniform:
                        try:
                            words = aligner.align_segment(segment, transcript, seg_lang)
                            intervals = TokenUniformAligner().spans_to_intervals(spans, transcript, words, channel=segment.channel, pre_padding_ms=self.config.masking.pre_padding_ms, post_padding_ms=self.config.masking.post_padding_ms)
                        except RequiredAlignmentError as exc:
                            # WhisperX produced no usable word alignments for this segment (often
                            # non-speech/music). Degrade to uniform timing instead of dropping the
                            # parent call; spans still get masked (fail-safe).
                            self.state.record_failure(f"{segment_id}:alignment", "segment", segment_id, f"whisperx_fallback_uniform: {exc!r}", retryable=True)
                            use_uniform = True
                        except Exception as exc:  # forced-alignment model load/runtime failure
                            # Degrade to uniform timing for THIS segment instead of crashing the
                            # stage; spans still get masked and the parent call is not dropped.
                            self.state.record_failure(f"{segment_id}:alignment", "segment", segment_id, f"whisperx_fallback_uniform: {exc!r}", retryable=True)
                            use_uniform = True
                    if use_uniform:
                        words = uniform_fallback.align(segment_id, transcript, segment.duration_sec)
                        intervals = uniform_fallback.spans_to_intervals(spans, transcript, words, channel=segment.channel, pre_padding_ms=self.config.masking.pre_padding_ms, post_padding_ms=self.config.masking.post_padding_ms)
                        degraded += 1
                else:
                    words = aligner.align(segment_id, transcript, segment.duration_sec)
                    intervals = aligner.spans_to_intervals(spans, transcript, words, channel=segment.channel, pre_padding_ms=self.config.masking.pre_padding_ms, post_padding_ms=self.config.masking.post_padding_ms)
                self.state.record_artifact(
                    f"alignment:{segment_id}",
                    "alignment",
                    Path(segment_id),
                    "completed",
                    {"segment_id": segment_id, "words": [dataclass_to_dict(w) for w in words], "intervals": [dataclass_to_dict(i) for i in intervals]},
                )
                self.state.clear_failure(f"{segment_id}:alignment")
                aligned += 1
            except RequiredAlignmentError as exc:
                failures += 1
                self.state.record_failure(f"{segment_id}:alignment", "segment", segment_id, repr(exc), retryable=True)
                self.state.record_artifact(f"alignment:{segment_id}", "alignment", Path(segment_id), "failed", {"segment_id": segment_id, "error": repr(exc)})
        flush_empty_alignments()
        summary = {"stage": "align", "aligned": aligned, "failed": failures, "degraded_uniform": degraded, "skipped_existing": skipped, "masked_whole_no_transcript": unreadable}
        self._write_report("align_summary.json", summary)
        return summary

    def mask_plan(self) -> dict[str, Any]:
        segment_by_id = {s.segment_id: s for s in self._load_segments()}
        files = {row["file_id"]: _audio_file_from_payload(row["payload"]) for row in self.state.fetch_files()}
        by_file: dict[str, list[dict[str, Any]]] = {file_id: [] for file_id in files}
        failed_file_ids: set[str] = set()
        existing_file_ids = {
            artifact["payload"].get("file_id") or artifact["artifact_id"].split(":", 1)[-1]
            for artifact in self.state.fetch_artifacts("mask_plan")
            if artifact.get("status") == "completed"
        }
        alignment_artifacts = self.state.fetch_artifacts("alignment")
        for artifact in iter_progress(alignment_artifacts, desc="Build mask plan", total=len(alignment_artifacts), unit="segment", enabled=self.config.progress_enabled):
            segment_id = artifact["payload"].get("segment_id")
            segment = segment_by_id.get(segment_id)
            if segment is None:
                continue
            if artifact["status"] != "completed":
                failed_file_ids.add(segment.file_id)
                continue
            for interval in artifact["payload"].get("intervals", []):
                full = dict(interval)
                full["start_sec"] = float(full["start_sec"]) + segment.start_sec
                full["end_sec"] = float(full["end_sec"]) + segment.start_sec
                full["segment_id"] = segment_id
                by_file.setdefault(segment.file_id, []).append(full)
        file_items = list(by_file.items())
        skipped = 0
        for file_id, intervals in iter_progress(file_items, desc="Save mask plans", total=len(file_items), unit="file", enabled=self.config.progress_enabled):
            if file_id in existing_file_ids:
                skipped += 1
                continue
            if file_id in failed_file_ids:
                self.state.record_artifact(f"mask_plan:{file_id}", "mask_plan", Path(file_id), "failed", {"file_id": file_id, "intervals": [], "error": "alignment_failed"})
                continue
            self.state.record_artifact(f"mask_plan:{file_id}", "mask_plan", Path(file_id), "completed", {"file_id": file_id, "intervals": intervals})
        summary = {"stage": "mask_plan", "files": len(by_file) - len(failed_file_ids) - skipped, "failed_files": len(failed_file_ids), "skipped_existing": skipped, "intervals": sum(len(v) for k, v in by_file.items() if k not in failed_file_ids and k not in existing_file_ids)}
        self._write_report("mask_plan_summary.json", summary)
        return summary

    def _redact_one(self, record: AudioFileRecord, intervals: list[MaskInterval]) -> tuple[Path, str, str | None, Any]:
        out_path = self.config.output_root / record.relative_path
        if not intervals:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_path, out_path)
            result = validate_audio_pair(record.source_path, out_path)
            status = "completed" if result.ok else "failed_validation"
            self._copy_sidecar(record.source_path, out_path)
            return out_path, status, None, result
        if record.source_path.suffix.lower() == ".wav":
            result = redact_wav_with_plan(record.source_path, out_path, intervals, strategy=self.config.masking.strategy)
            status = "completed" if result.ok else "failed_validation"
            self._copy_sidecar(record.source_path, out_path)
            return out_path, status, None, result
        decoded = temp_wav_path("amc_decode", base_dir=self.config.temp_dir)
        masked = temp_wav_path("amc_masked", base_dir=self.config.temp_dir)
        try:
            decode_to_wav(record.source_path, decoded)
            result = redact_wav_with_plan(decoded, masked, intervals, strategy=self.config.masking.strategy)
            final_path = out_path
            fallback_error = None
            try:
                encode_from_wav(masked, out_path, source_path=record.source_path)
                status = "completed" if result.ok else "failed_validation"
            except Exception as encode_exc:
                if not self.config.masking.allow_wav_fallback:
                    raise
                fallback_error = repr(encode_exc)
                final_path = out_path.with_suffix(".wav")
                encode_from_wav(masked, final_path)
                status = "completed_with_fallback" if result.ok else "failed_validation"
        finally:
            shutil.rmtree(decoded.parent, ignore_errors=True)
            shutil.rmtree(masked.parent, ignore_errors=True)
        self._copy_sidecar(record.source_path, final_path)
        return final_path, status, fallback_error, result

    def _copy_sidecar(self, source_path: Path, output_path: Path) -> None:
        if not self.config.copy_sidecars:
            return
        sidecar = source_path.parent / "metadata.json"
        if not sidecar.exists():
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, output_path.parent / "metadata.json")
        except Exception:
            pass

    def redact(self) -> dict[str, Any]:
        files = {row["file_id"]: _audio_file_from_payload(row["payload"]) for row in self.state.iter_files()}
        outputs = []
        failures = []
        skipped = 0
        existing_redacted = {
            artifact["payload"].get("file_id"): artifact
            for artifact in self.state.iter_artifacts("redacted")
            if artifact.get("status") in {"completed", "completed_with_fallback"}
        }
        pending: list[tuple[str, AudioFileRecord, list[MaskInterval]]] = []
        for artifact in self.state.iter_artifacts("mask_plan"):
            if artifact["status"] != "completed":
                continue
            file_id = artifact["payload"].get("file_id")
            record = files.get(file_id)
            if record is None:
                continue
            existing = existing_redacted.get(file_id)
            if existing and Path(existing["payload"].get("path", "")).exists():
                skipped += 1
                outputs.append(existing["payload"].get("path", ""))
                continue
            intervals = [MaskInterval(**{k: v for k, v in raw.items() if k in {"channel", "start_sec", "end_sec", "reason", "entity_type", "confidence", "source"}}) for raw in artifact["payload"].get("intervals", [])]
            pending.append((file_id, record, intervals))

        # Redaction decodes whole calls into RAM; concurrency is gated by memory,
        # not CPU. resolved_redact_workers() scales by MemAvailable / per-call peak
        # so big-RAM boxes use their headroom while 16 GB boxes stay OOM-safe.
        workers = self.config.resolved_redact_workers()

        def persist(file_id: str, record: AudioFileRecord, final_path: Path, status: str, fallback_error: str | None, result: Any) -> None:
            out_path = self.config.output_root / record.relative_path
            self.state.record_artifact(
                f"redacted:{file_id}",
                "redacted",
                final_path,
                status,
                {"file_id": file_id, "path": str(final_path), "preferred_path": str(out_path), "fallback_error": fallback_error, "validation": dataclass_to_dict(result)},
            )
            self.state.clear_failure(f"{file_id}:redact")
            outputs.append(str(final_path))

        if workers <= 1 or len(pending) <= 1:
            for file_id, record, intervals in iter_progress(pending, desc="Redact audio", total=len(pending), unit="file", enabled=self.config.progress_enabled):
                try:
                    final_path, status, fallback_error, result = self._redact_one(record, intervals)
                    persist(file_id, record, final_path, status, fallback_error, result)
                except Exception as exc:
                    failures.append({"file_id": file_id, "error": repr(exc)})
                    self.state.record_failure(f"{file_id}:redact", "file", file_id, repr(exc), retryable=True)
        else:
            from concurrent.futures import ThreadPoolExecutor

            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for file_id, record, intervals in pending:
                    futures[pool.submit(self._redact_one, record, intervals)] = (file_id, record)
                for future in iter_progress(futures, desc="Redact audio", total=len(futures), unit="file", enabled=self.config.progress_enabled):
                    file_id, record = futures[future]
                    try:
                        final_path, status, fallback_error, result = future.result()
                        persist(file_id, record, final_path, status, fallback_error, result)
                    except Exception as exc:
                        failures.append({"file_id": file_id, "error": repr(exc)})
                        self.state.record_failure(f"{file_id}:redact", "file", file_id, repr(exc), retryable=True)
        summary = {"stage": "redact", "outputs": outputs, "failures": failures, "skipped_existing": skipped}
        self._write_report("redact_summary.json", summary)
        return summary

    def validate(self) -> dict[str, Any]:
        files = {row["file_id"]: _audio_file_from_payload(row["payload"]) for row in self.state.fetch_files()}
        validations = []
        redacted_artifacts = self.state.fetch_artifacts("redacted")
        for artifact in iter_progress(redacted_artifacts, desc="Validate outputs", total=len(redacted_artifacts), unit="file", enabled=self.config.progress_enabled):
            file_id = artifact["payload"].get("file_id")
            record = files.get(file_id)
            if record is None:
                continue
            output = Path(artifact["payload"]["path"])
            result = validate_audio_pair(record.source_path, output)
            validations.append(dataclass_to_dict(result))
            self.state.record_artifact(f"validation:{file_id}", "validation", output, result.status, dataclass_to_dict(result))
        summary = {"stage": "validate", "validations": validations}
        self._write_report("validation_summary.json", summary)
        return summary

    def write_manifests(self, segments=None) -> dict[str, Any]:
        # Stream segments straight from the state store (the durable source of truth) so the
        # manifest stage never materializes the full row list or a pandas DataFrame. extra_rows
        # is the in-memory join table (transcripts/pii/mask/redacted by segment id); the row
        # values themselves are produced lazily and flushed in Parquet row-group batches.
        seg_iter = self._iter_segment_records() if segments is None else segments
        paths = write_segment_manifests(
            self.config.output_root,
            seg_iter,
            self._manifest_extras(),
            enable_dataframe_exports=self.config.manifest_dataframe_exports,
            xlsx_max_rows=self.config.manifest_xlsx_max_rows,
            write_csv=self.config.manifest_write_csv,
            write_per_year=self.config.manifest_write_per_year,
            parquet_batch_size=self.config.manifest_parquet_batch_size,
        )
        return {"paths": [str(p) for p in paths]}

    def pause(self, worker_id: str | None = None, global_pause: bool = False) -> dict[str, Any]:
        self.state.request_pause(self.config.run_id, worker_id=worker_id, global_pause=global_pause)
        result = {"run_id": self.config.run_id, "worker_id": worker_id, "global": global_pause}
        self._write_report("pause_request.json", result)
        return result

    def clean_cache(self) -> dict[str, Any]:
        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        return {"cache_dir": str(self.config.cache_dir), "status": "cleaned"}

    def _write_report(self, name: str, payload: dict[str, Any]) -> Path:
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.reports_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def _load_segments(self) -> list[SegmentRecord]:
        return list(self._iter_segment_records())

    def _iter_segment_records(self):
        # Streaming variant of _load_segments: yields SegmentRecord without materializing the
        # full list, so the manifest stage stays memory-bounded at millions of rows.
        for row in self.state.iter_segments():
            payload = dict(row["payload"])
            payload["status"] = row.get("status", payload.get("status", "preprocessed"))
            yield _segment_from_payload(payload)

    def _manifest_extras(self) -> dict[str, dict[str, Any]]:
        extras: dict[str, dict[str, Any]] = {}
        file_metadata: dict[str, dict[str, Any]] = {
            row["file_id"]: (row.get("payload") or {}).get("audio_metadata") or {}
            for row in self.state.iter_files()
        }
        file_redacted = {
            artifact["payload"].get("file_id"): artifact
            for artifact in self.state.iter_artifacts("redacted")
            if artifact["payload"].get("file_id")
        }
        # (call_id, channel) -> speaker_cluster_id. Prefer loading the global mapping produced by
        # cluster-speakers directly from clusters.parquet -- this avoids the speaker_assign stage
        # persisting one artifact per side into FSx-backed SQLite (a write storm that stalled
        # workers). Fall back to per-side artifacts for back-compat if the file isn't configured.
        # Empty when neither source is available, so the column is simply absent (back-compat).
        speaker_map: dict[tuple[str, str], str] = {}
        speaker_clusters_loaded = False
        speaker_clusters_path = self.config.speaker_clusters_path
        if speaker_clusters_path and Path(speaker_clusters_path).exists():
            speaker_map = load_cluster_map(Path(speaker_clusters_path))
            speaker_clusters_loaded = True
        else:
            speaker_map = {
                (str(a["payload"].get("call_id", "")), str(a["payload"].get("channel", ""))): a["payload"].get("speaker_cluster_id", "")
                for a in self.state.iter_artifacts("speaker_cluster")
            }
        for row in self.state.iter_model_results():
            segment_id = row["segment_id"]
            payload = row["payload"]
            model = row["model_name"]
            extras.setdefault(segment_id, {})
            extras[segment_id][f"{model}_transcript"] = payload.get("transcript", "")
            extras[segment_id][f"{model}_normalized_transcript"] = payload.get("normalized_transcript", "")
            extras[segment_id][f"{model}_confidence"] = payload.get("confidence", "")
            extras[segment_id][f"{model}_error"] = payload.get("error", "")
            if model in LANGUAGE_DETECTING_MODELS:
                extras[segment_id]["language"] = payload.get("language") or extras[segment_id].get("language", "")
                extras[segment_id]["language_confidence"] = payload.get("language_confidence") or payload.get("confidence") or extras[segment_id].get("language_confidence", "")
        for artifact in self.state.iter_artifacts("consensus"):
            payload = artifact["payload"]
            segment_id = payload.get("segment_id")
            extras.setdefault(segment_id, {}).update(
                {
                    "final_transcript": payload.get("final_transcript", ""),
                    "normalized_final_transcript": payload.get("normalized_transcript", ""),
                    "consensus_method": payload.get("method", ""),
                    "consensus_score": payload.get("score", ""),
                    "consensus_strong": payload.get("strong", ""),
                    "selected_model": payload.get("selected_model", ""),
                }
            )
        for artifact in self.state.iter_artifacts("pii"):
            payload = artifact["payload"]
            segment_id = payload.get("segment_id")
            spans = payload.get("spans", [])
            extras.setdefault(segment_id, {})["pii_spans_json"] = json.dumps(spans, sort_keys=True, default=str)
            extras[segment_id]["pii_count"] = len(spans)
        for artifact in self.state.iter_artifacts("alignment"):
            payload = artifact["payload"]
            segment_id = payload.get("segment_id")
            intervals = payload.get("intervals", [])
            extras.setdefault(segment_id, {})["mask_intervals_json"] = json.dumps(intervals, sort_keys=True, default=str)
            extras[segment_id]["alignment_status"] = artifact["status"]
        # Join file-level metadata + redacted artifacts onto each segment. Stream the segment
        # rows (segment_id + file_id only) instead of materializing a {segment_id: SegmentRecord}
        # map -- the latter was a second full in-memory copy of every segment.
        for row in self.state.iter_segments():
            segment_id = row["segment_id"]
            payload = row.get("payload") or {}
            file_id = row.get("file_id") or payload.get("file_id")
            if speaker_map or speaker_clusters_loaded:
                key = (str(payload.get("call_id", "")), str(int(payload.get("channel", 0))))
                cid = speaker_map.get(key, "")
                if not cid and speaker_clusters_loaded:
                    # Side had no usable audio to embed -> deterministic per-side fallback id so
                    # every segment is still labeled and never accidentally grouped with others.
                    cid = f"spk_fallback_{key[0]}_ch{key[1]}"
                extras.setdefault(segment_id, {})["speaker_cluster_id"] = cid
            meta = file_metadata.get(file_id)
            if meta:
                extras.setdefault(segment_id, {})
                extras[segment_id]["source_codec"] = meta.get("codec", "")
                extras[segment_id]["source_sample_rate"] = meta.get("sample_rate", "")
                extras[segment_id]["source_channels"] = meta.get("channels", "")
                extras[segment_id]["source_duration_sec"] = meta.get("duration_sec", "")
                extras[segment_id]["source_bitrate"] = meta.get("bitrate", "")
                extras[segment_id]["source_channel_layout"] = meta.get("channel_layout", "")
            redacted = file_redacted.get(file_id)
            if not redacted:
                continue
            redacted_path = Path(redacted["payload"].get("path", ""))
            extras.setdefault(segment_id, {})
            extras[segment_id]["redacted_audio_path_abs"] = str(redacted_path)
            extras[segment_id]["redacted_audio_path_rel"] = _safe_relative(redacted_path, self.config.output_root)
            extras[segment_id]["redacted_status"] = redacted["status"]
            extras[segment_id]["redacted_fallback_error"] = redacted["payload"].get("fallback_error") or ""
        panel = self._agreement_panel()
        for ex in extras.values():
            _annotate_agreement(ex, panel)
        return extras

    def _agreement_panel(self) -> list[str]:
        """The ASR models this run votes with, in tie-break order.

        Derived from the run's own enabled models so that swapping one model for
        another (whisper -> parakeet) keeps the panel at four and the
        `all_4_agree` / `incomplete_N_models` labels directly comparable across runs.
        """
        enabled = {name for name, cfg in (self.config.asr_models or {}).items() if cfg.enabled}
        panel = [m for m in DEFAULT_PRIORITY if m in enabled]
        return panel or list(DEFAULT_AGREEMENT_MODELS)


def _usable_norms(ex, panel: list[str]):
    out = []
    for m in panel:
        if str(ex.get(m + "_error") or "").strip():
            continue
        if not str(ex.get(m + "_transcript") or "").strip():
            continue
        out.append(str(ex.get(m + "_normalized_transcript") or "").strip())
    return out


def _agreement_label(present, agreeing, panel_size: int):
    if present < panel_size:
        return "incomplete_" + str(present) + "_models"
    return {4: "all_4_agree", 3: "three_agree", 2: "two_agree"}.get(agreeing, "all_different")


def _annotate_agreement(ex: dict[str, Any], panel: list[str]) -> None:
    norms = _usable_norms(ex, panel)
    present = len(norms)
    agreeing = max(Counter(norms).values()) if norms else 0
    ex["model_agreement"] = _agreement_label(present, agreeing, len(panel))
    ex["models_present"] = present
    ex["models_agreeing"] = agreeing


def _segment_from_payload(payload: dict[str, Any]) -> SegmentRecord:
    data = dict(payload)
    data["source_path"] = Path(data["source_path"])
    data["segment_audio_path"] = Path(data["segment_audio_path"])
    return SegmentRecord(**{k: data[k] for k in SegmentRecord.__dataclass_fields__ if k in data})


def _compact_audio_metadata(meta: Any) -> dict[str, Any]:
    return {
        "codec": getattr(meta, "codec", None),
        "container": getattr(meta, "container", None),
        "duration_sec": getattr(meta, "duration_sec", None),
        "sample_rate": getattr(meta, "sample_rate", None),
        "channels": getattr(meta, "channels", None),
        "bitrate": getattr(meta, "bitrate", None),
        "channel_layout": getattr(meta, "channel_layout", None),
        "format_name": getattr(meta, "format_name", None),
    }


def _audio_file_from_payload(payload: dict[str, Any]) -> AudioFileRecord:
    data = dict(payload)
    data["source_path"] = Path(data["source_path"])
    data["relative_path"] = Path(data["relative_path"])
    return AudioFileRecord(**{k: data[k] for k in AudioFileRecord.__dataclass_fields__ if k in data})


def _pii_span_from_payload(payload: dict[str, Any]) -> PIISpan:
    return PIISpan(**{k: payload[k] for k in PIISpan.__dataclass_fields__ if k in payload})


def _dedupe_pii(spans: list[PIISpan]) -> list[PIISpan]:
    best: dict[tuple[int, int, str], PIISpan] = {}
    for span in spans:
        key = (span.start_char, span.end_char, span.entity_type)
        if key not in best or span.confidence > best[key].confidence:
            best[key] = span
    return sorted(best.values(), key=lambda x: (x.start_char, x.end_char))


def _map_span_to_final(span: PIISpan, source_transcript: str, final_transcript: str) -> PIISpan | None:
    if source_transcript == final_transcript:
        return span
    haystack = final_transcript.lower()
    needle = span.text.lower().strip()
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return PIISpan(
        entity_type=span.entity_type,
        text=final_transcript[idx : idx + len(span.text)],
        start_char=idx,
        end_char=idx + len(span.text),
        confidence=span.confidence,
        source=span.source,
        raw={**span.raw, "source_transcript": source_transcript, "source_start_char": span.start_char, "source_end_char": span.end_char},
    )


def _expand_numeric_pii_context(span: PIISpan, transcript: str) -> PIISpan:
    if not _is_numeric_pii_candidate(span):
        return span
    text = str(transcript or "")
    left = max(0, span.start_char)
    right = min(len(text), span.end_char)
    allowed = set("0123456789+().-/ #")
    while left > 0 and text[left - 1] in allowed:
        left -= 1
    while right < len(text) and text[right] in allowed:
        right += 1
    while left < right and text[left] in " +().-/ #":
        left += 1
    while right > left and text[right - 1] in " +().-/ #":
        right -= 1
    expanded = text[left:right]
    if _digit_count(expanded) < max(4, _digit_count(span.text)):
        return span
    entity_type = "PHONE" if _digit_count(expanded) >= 7 and span.entity_type.upper() in {"NUMBER", "CARDINAL", "PHONE", "PHONE_NUMBER"} else span.entity_type
    return PIISpan(
        entity_type=entity_type,
        text=expanded,
        start_char=left,
        end_char=right,
        confidence=span.confidence,
        source=span.source,
        raw={**span.raw, "expanded_from_text": span.text, "expanded_from_start_char": span.start_char, "expanded_from_end_char": span.end_char},
    )


def _is_numeric_pii_candidate(span: PIISpan) -> bool:
    entity = span.entity_type.upper()
    return entity in {
        "PHONE",
        "PHONE_NUMBER",
        "SSN",
        "ID",
        "ACCOUNT",
        "ACCOUNT_NUMBER",
        "POLICY",
        "POLICY_NUMBER",
        "MEMBER_ID",
        "SUBSCRIBER_ID",
        "ZIP",
        "DATE",
        "DOB",
        "NUMBER",
        "CARDINAL",
    } or _digit_count(span.text) >= 4


def _digit_count(text: str) -> int:
    return len(re.findall(r"\d", str(text or "")))


def _dedupe_pii_dicts(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for span in spans:
        key = (int(span["start_char"]), int(span["end_char"]), str(span["entity_type"]), str(span.get("transcript_source", "")))
        if key not in best or float(span.get("confidence") or 0.0) > float(best[key].get("confidence") or 0.0):
            best[key] = span
    return sorted(best.values(), key=lambda x: (int(x["start_char"]), int(x["end_char"]), str(x.get("transcript_source", ""))))


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)
