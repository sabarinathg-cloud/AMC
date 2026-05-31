from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from .alignment import RequiredAlignmentError, TokenUniformAligner, TorchaudioCTCAligner, WhisperXAligner
from .audio import MissingDependencyError, decode_to_wav, encode_from_wav, temp_wav_path
from .config import PipelineConfig
from .consensus import build_consensus
from .discovery import discover_audio_files
from .inspection import inspect_audio
from .manifests import write_segment_manifests
from .masking import redact_wav_with_plan
from .models import ASRResult, AudioFileRecord, MaskInterval, PIISpan, SegmentRecord, dataclass_to_dict
from .normalization import normalize_transcript
from .pii_detection import build_enabled_detectors, preflight_detectors
from .preprocessing import preprocess_file
from .progress import iter_progress
from .resources import detect_resources
from .segmentation import preflight_vad_backend
from .state import PostgresStateStore, SQLiteStateStore
from .transcription import build_enabled_adapters
from .validation import validate_audio_pair


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
                self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "inspected", "payload": dataclass_to_dict(record)})
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
            return self.write_manifests(self._load_segments())
        raise ValueError(f"Unsupported stage: {stage}")

    def preprocess(self) -> dict[str, Any]:
        records = discover_audio_files(self.config)
        all_segments = []
        errors = []
        for record in iter_progress(records, desc="Preprocess audio", total=len(records), unit="file", enabled=self.config.progress_enabled):
            if self.state.should_pause(self.config.run_id):
                break
            try:
                self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "preprocessing", "payload": dataclass_to_dict(record)})
                segments = preprocess_file(record, self.config)
                all_segments.extend(segments)
                for segment in segments:
                    self.state.upsert_segment(segment.segment_id, record.file_id, "preprocessed", dataclass_to_dict(segment))
                self.state.upsert_file({"file_id": record.file_id, "source_path": str(record.source_path), "status": "preprocessed", "payload": dataclass_to_dict(record)})
            except Exception as exc:
                errors.append({"file_id": record.file_id, "source_path": str(record.source_path), "error": repr(exc)})
                self.state.record_failure(f"{record.file_id}:preprocess", "file", record.file_id, repr(exc), retryable=True)
        manifest_paths = write_segment_manifests(self.config.output_root, all_segments) if all_segments else []
        summary = {"stage": "preprocess", "files": len(records), "segments": len(all_segments), "errors": errors, "manifests": [str(p) for p in manifest_paths]}
        self._write_report("preprocess_summary.json", summary)
        return summary

    def asr(self) -> dict[str, Any]:
        segments = self._load_segments()
        adapters = build_enabled_adapters(self.config)
        counts: dict[str, int] = {}
        failed_models: list[dict[str, Any]] = []
        supported = {x.lower() for x in self.config.languages}
        language_by_segment: dict[str, str | None] = {}
        language_confidence_by_segment: dict[str, float | None] = {}
        segment_by_id = {s.segment_id: s for s in segments}
        adapters = sorted(adapters, key=lambda a: 0 if a.name == "whisper" else 1)
        for adapter in adapters:
            adapter.progress_enabled = self.config.progress_enabled
        for adapter in iter_progress(adapters, desc="ASR models", total=len(adapters), unit="model", enabled=self.config.progress_enabled):
            model_segments = segments
            if adapter.name != "whisper" and language_by_segment:
                model_segments = [
                    replace(
                        s,
                        language=language_by_segment.get(s.segment_id),
                        language_confidence=language_confidence_by_segment.get(s.segment_id),
                    )
                    for s in segments
                    if (language_by_segment.get(s.segment_id) or "").lower() in supported
                ]
            try:
                adapter.preflight()
                results = adapter.transcribe_batch(model_segments)
            except Exception as exc:
                tb = traceback.format_exc()
                failed_models.append({"model": adapter.name, "error": repr(exc), "segments": len(model_segments)})
                self.state.record_failure(f"asr:{adapter.name}", "model", adapter.name, repr(exc), retryable=True, traceback=tb)
                results = [
                    ASRResult(
                        segment_id=s.segment_id,
                        model_name=adapter.name,
                        transcript="",
                        confidence=0.0,
                        language=s.language,
                        language_confidence=s.language_confidence,
                        error=repr(exc),
                        raw={"model_stage_failure": True},
                    )
                    for s in model_segments
                ]
            counts[adapter.name] = len(results)
            for result in results:
                if adapter.name == "whisper":
                    language_by_segment[result.segment_id] = result.language
                    language_confidence_by_segment[result.segment_id] = result.language_confidence
                    segment = segment_by_id.get(result.segment_id)
                    if segment is not None:
                        payload = dataclass_to_dict(replace(segment, language=result.language, language_confidence=result.language_confidence))
                        self.state.upsert_segment(segment.segment_id, segment.file_id, segment.status, payload)
                self.state.upsert_model_result(result.segment_id, result.model_name, "failed" if result.error else "transcribed", dataclass_to_dict(result))
        skipped = []
        for segment in segments:
            lang = language_by_segment.get(segment.segment_id)
            if lang and lang.lower() not in supported:
                skipped.append({"segment_id": segment.segment_id, "language": lang, "reason": "unsupported_language"})
                payload = dataclass_to_dict(segment)
                payload["language"] = lang
                payload["status_reason"] = "unsupported_language"
                self.state.upsert_segment(segment.segment_id, segment.file_id, "skipped_unsupported_language", payload)
        summary = {"stage": "asr", "segments": len(segments), "models": counts, "model_failures": failed_models, "skipped_unsupported_language": skipped}
        self._write_report("asr_summary.json", summary)
        return summary

    def normalize(self) -> dict[str, Any]:
        rows = self.state.fetch_model_results()
        count = 0
        for row in iter_progress(rows, desc="Normalize transcripts", total=len(rows), unit="result", enabled=self.config.progress_enabled):
            payload = row["payload"]
            payload["normalized_transcript"] = normalize_transcript(payload.get("transcript", ""), remove_fillers=True)
            self.state.upsert_model_result(row["segment_id"], row["model_name"], row["status"], payload)
            count += 1
        summary = {"stage": "normalize", "model_results": count}
        self._write_report("normalize_summary.json", summary)
        return summary

    def consensus(self) -> dict[str, Any]:
        grouped: dict[str, list[ASRResult]] = {}
        for row in self.state.fetch_model_results():
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
        items = list(grouped.items())
        for segment_id, results in iter_progress(items, desc="Build consensus", total=len(items), unit="segment", enabled=self.config.progress_enabled):
            result = build_consensus(results, min_successful_models=self.config.consensus_min_models)
            if result.strong:
                strong += 1
            self.state.record_artifact(f"consensus:{segment_id}", "consensus", Path(segment_id), "completed", dataclass_to_dict(result))
        summary = {"stage": "consensus", "segments": len(grouped), "strong": strong, "weak": len(grouped) - strong}
        self._write_report("consensus_summary.json", summary)
        return summary

    def pii(self) -> dict[str, Any]:
        detectors = build_enabled_detectors(self.config)
        preflight_detectors(detectors)
        artifacts = self.state.fetch_artifacts("consensus")
        count = 0
        for artifact in iter_progress(artifacts, desc="Detect PII", total=len(artifacts), unit="segment", enabled=self.config.progress_enabled):
            payload = artifact["payload"]
            final_transcript = payload.get("final_transcript", "")
            span_dicts = self._detect_pii_for_segment(payload.get("segment_id") or artifact["artifact_id"].split(":", 1)[-1], final_transcript, detectors)
            segment_id = payload.get("segment_id") or artifact["artifact_id"].split(":", 1)[-1]
            self.state.record_artifact(f"pii:{segment_id}", "pii", Path(segment_id), "completed", {"segment_id": segment_id, "spans": span_dicts})
            count += len(span_dicts)
        summary = {"stage": "pii", "segments": len(artifacts), "spans": count}
        self._write_report("pii_summary.json", summary)
        return summary

    def _detect_pii_for_segment(self, segment_id: str, final_transcript: str, detectors) -> list[dict[str, Any]]:
        candidates: list[tuple[str, str]] = [("final", final_transcript)]
        if self.config.pii_on_all_model_transcripts:
            for row in self.state.fetch_model_results(segment_id):
                text = row["payload"].get("transcript", "")
                if text:
                    candidates.append((row["model_name"], text))
        span_dicts: list[dict[str, Any]] = []
        for source_name, transcript in candidates:
            for detector in detectors:
                for span in detector.detect(transcript):
                    mapped = _map_span_to_final(span, transcript, final_transcript)
                    if mapped is None:
                        continue
                    data = dataclass_to_dict(mapped)
                    data["transcript_source"] = source_name
                    data["detector_source"] = span.source
                    span_dicts.append(data)
        return _dedupe_pii_dicts(span_dicts)

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
        aligned = 0
        failures = 0
        pii_artifacts = self.state.fetch_artifacts("pii")
        for artifact in iter_progress(pii_artifacts, desc="Force alignment", total=len(pii_artifacts), unit="segment", enabled=self.config.progress_enabled):
            payload = artifact["payload"]
            segment_id = payload["segment_id"]
            segment = segment_by_id.get(segment_id)
            consensus = consensus_by_segment.get(segment_id)
            if segment is None or consensus is None:
                continue
            spans = [_pii_span_from_payload(span) for span in payload.get("spans", [])]
            if not spans:
                self.state.record_artifact(f"alignment:{segment_id}", "alignment", Path(segment_id), "completed", {"segment_id": segment_id, "words": [], "intervals": []})
                continue
            try:
                transcript = consensus.get("final_transcript", "")
                if isinstance(aligner, WhisperXAligner):
                    words = aligner.align_segment(segment, transcript, segment.language or consensus.get("language") or "en")
                    intervals = TokenUniformAligner().spans_to_intervals(spans, transcript, words, channel=segment.channel, pre_padding_ms=self.config.masking.pre_padding_ms, post_padding_ms=self.config.masking.post_padding_ms)
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
                aligned += 1
            except RequiredAlignmentError as exc:
                failures += 1
                self.state.record_failure(f"{segment_id}:alignment", "segment", segment_id, repr(exc), retryable=True)
                self.state.record_artifact(f"alignment:{segment_id}", "alignment", Path(segment_id), "failed", {"segment_id": segment_id, "error": repr(exc)})
        summary = {"stage": "align", "aligned": aligned, "failed": failures}
        self._write_report("align_summary.json", summary)
        return summary

    def mask_plan(self) -> dict[str, Any]:
        segment_by_id = {s.segment_id: s for s in self._load_segments()}
        files = {row["file_id"]: _audio_file_from_payload(row["payload"]) for row in self.state.fetch_files()}
        by_file: dict[str, list[dict[str, Any]]] = {file_id: [] for file_id in files}
        failed_file_ids: set[str] = set()
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
        for file_id, intervals in iter_progress(file_items, desc="Save mask plans", total=len(file_items), unit="file", enabled=self.config.progress_enabled):
            if file_id in failed_file_ids:
                self.state.record_artifact(f"mask_plan:{file_id}", "mask_plan", Path(file_id), "failed", {"file_id": file_id, "intervals": [], "error": "alignment_failed"})
                continue
            self.state.record_artifact(f"mask_plan:{file_id}", "mask_plan", Path(file_id), "completed", {"file_id": file_id, "intervals": intervals})
        summary = {"stage": "mask_plan", "files": len(by_file) - len(failed_file_ids), "failed_files": len(failed_file_ids), "intervals": sum(len(v) for k, v in by_file.items() if k not in failed_file_ids)}
        self._write_report("mask_plan_summary.json", summary)
        return summary

    def redact(self) -> dict[str, Any]:
        files = {row["file_id"]: _audio_file_from_payload(row["payload"]) for row in self.state.fetch_files()}
        outputs = []
        failures = []
        mask_artifacts = self.state.fetch_artifacts("mask_plan")
        for artifact in iter_progress(mask_artifacts, desc="Redact audio", total=len(mask_artifacts), unit="file", enabled=self.config.progress_enabled):
            if artifact["status"] != "completed":
                continue
            file_id = artifact["payload"].get("file_id")
            record = files.get(file_id)
            if record is None:
                continue
            intervals = [MaskInterval(**{k: v for k, v in raw.items() if k in {"channel", "start_sec", "end_sec", "reason", "entity_type", "confidence", "source"}}) for raw in artifact["payload"].get("intervals", [])]
            out_path = self.config.output_root / record.relative_path
            try:
                if not intervals:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(record.source_path, out_path)
                    final_path = out_path
                    fallback_error = None
                    result = validate_audio_pair(record.source_path, final_path)
                    status = "completed" if result.ok else "failed_validation"
                elif record.source_path.suffix.lower() == ".wav":
                    result = redact_wav_with_plan(record.source_path, out_path, intervals, strategy=self.config.masking.strategy)
                    status = "completed" if result.ok else "failed_validation"
                    final_path = out_path
                    fallback_error = None
                else:
                    decoded = temp_wav_path("amc_decode")
                    masked = temp_wav_path("amc_masked")
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
                self.state.record_artifact(
                    f"redacted:{file_id}",
                    "redacted",
                    final_path,
                    status,
                    {"file_id": file_id, "path": str(final_path), "preferred_path": str(out_path), "fallback_error": fallback_error, "validation": dataclass_to_dict(result)},
                )
                outputs.append(str(final_path))
            except Exception as exc:
                failures.append({"file_id": file_id, "error": repr(exc)})
                self.state.record_failure(f"{file_id}:redact", "file", file_id, repr(exc), retryable=True)
        summary = {"stage": "redact", "outputs": outputs, "failures": failures}
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

    def write_manifests(self, segments) -> dict[str, Any]:
        paths = write_segment_manifests(self.config.output_root, segments, self._manifest_extras())
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
        segments = []
        for row in self.state.fetch_segments():
            payload = dict(row["payload"])
            payload["status"] = row.get("status", payload.get("status", "preprocessed"))
            segments.append(_segment_from_payload(payload))
        return segments

    def _manifest_extras(self) -> dict[str, dict[str, Any]]:
        extras: dict[str, dict[str, Any]] = {s.segment_id: {} for s in self._load_segments()}
        segment_by_id = {s.segment_id: s for s in self._load_segments()}
        file_redacted = {
            artifact["payload"].get("file_id"): artifact
            for artifact in self.state.fetch_artifacts("redacted")
            if artifact["payload"].get("file_id")
        }
        for row in self.state.fetch_model_results():
            segment_id = row["segment_id"]
            payload = row["payload"]
            model = row["model_name"]
            extras.setdefault(segment_id, {})
            extras[segment_id][f"{model}_transcript"] = payload.get("transcript", "")
            extras[segment_id][f"{model}_normalized_transcript"] = payload.get("normalized_transcript", "")
            extras[segment_id][f"{model}_error"] = payload.get("error", "")
            if model == "whisper":
                extras[segment_id]["language"] = payload.get("language") or extras[segment_id].get("language", "")
                extras[segment_id]["language_confidence"] = payload.get("language_confidence") or payload.get("confidence") or extras[segment_id].get("language_confidence", "")
        for artifact in self.state.fetch_artifacts("consensus"):
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
        for artifact in self.state.fetch_artifacts("pii"):
            payload = artifact["payload"]
            segment_id = payload.get("segment_id")
            spans = payload.get("spans", [])
            extras.setdefault(segment_id, {})["pii_spans_json"] = json.dumps(spans, sort_keys=True, default=str)
            extras[segment_id]["pii_count"] = len(spans)
        for artifact in self.state.fetch_artifacts("alignment"):
            payload = artifact["payload"]
            segment_id = payload.get("segment_id")
            intervals = payload.get("intervals", [])
            extras.setdefault(segment_id, {})["mask_intervals_json"] = json.dumps(intervals, sort_keys=True, default=str)
            extras[segment_id]["alignment_status"] = artifact["status"]
        for segment_id, segment in segment_by_id.items():
            redacted = file_redacted.get(segment.file_id)
            if not redacted:
                continue
            redacted_path = Path(redacted["payload"].get("path", ""))
            extras.setdefault(segment_id, {})
            extras[segment_id]["redacted_audio_path_abs"] = str(redacted_path)
            extras[segment_id]["redacted_audio_path_rel"] = _safe_relative(redacted_path, self.config.output_root)
            extras[segment_id]["redacted_status"] = redacted["status"]
            extras[segment_id]["redacted_fallback_error"] = redacted["payload"].get("fallback_error") or ""
        return extras


def _segment_from_payload(payload: dict[str, Any]) -> SegmentRecord:
    data = dict(payload)
    data["source_path"] = Path(data["source_path"])
    data["segment_audio_path"] = Path(data["segment_audio_path"])
    return SegmentRecord(**{k: data[k] for k in SegmentRecord.__dataclass_fields__ if k in data})


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
