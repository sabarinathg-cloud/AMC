import csv
import json
import math
import os
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from amc_pipeline.alignment import TokenUniformAligner
from amc_pipeline.audio import decode_to_wav
from amc_pipeline.cli import build_parser, load_cli_config
from amc_pipeline.config import PipelineConfig
from amc_pipeline.consensus import build_consensus
from amc_pipeline.discovery import discover_audio_files
from amc_pipeline.inspection import inspect_audio
from amc_pipeline.masking import redact_wav_with_plan
from amc_pipeline.models import ASRResult, AlignmentWord, MaskInterval, PIISpan
from amc_pipeline.normalization import normalize_transcript
from amc_pipeline.pii_detection import RegexPIIDetector
from amc_pipeline.pipeline import Pipeline, _expand_numeric_pii_context
from amc_pipeline.preprocessing import preprocess_file
from amc_pipeline.segmentation import split_chunks_on_pauses
from amc_pipeline.state import SQLiteStateStore
from amc_pipeline.transcription import _require_qwen_model_config, _require_qwen_runtime_versions
from amc_pipeline.models import dataclass_to_dict


class FailingASRAdapter:
    name = "qwen"
    progress_enabled = False
    closed = False

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        raise RuntimeError("model boom")

    def close(self):
        type(self).closed = True


class StaticASRAdapter:
    name = "granite"
    progress_enabled = False
    closed = False

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        return [ASRResult(s.segment_id, self.name, "safe transcript", 0.9, s.language or "en") for s in segments]

    def close(self):
        type(self).closed = True


def write_stereo_wav(path: Path, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(nframes):
            t = i / sample_rate
            left = int(12000 * math.sin(2 * math.pi * 440 * t))
            right = int(9000 * math.sin(2 * math.pi * 880 * t))
            if 0.45 < t < 0.55:
                left = 0
                right = 0
            frames.extend(struct.pack("<hh", left, right))
        wf.writeframes(bytes(frames))


def read_wav_ints(path: Path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    rows = [values[i : i + channels] for i in range(0, len(values), channels)]
    return sample_rate, channels, rows


class CorePipelineTests(unittest.TestCase):
    def test_discovery_inspection_call_id_sidecar_and_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            first = root / "2022" / "call123" / "audio.wav"
            duplicate = root / "2023" / "callDup" / "audio.wav"
            write_stereo_wav(first)
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes(first.read_bytes())
            (first.parent / "metadata.json").write_text(json.dumps({"customer": {"id": "C1"}}))

            config = PipelineConfig(input_root=root, output_root=Path(td) / "output")
            records = discover_audio_files(config)

            self.assertEqual(len(records), 2)
            by_call = {r.call_id: r for r in records}
            self.assertEqual(by_call["call123"].year, "2022")
            self.assertEqual(by_call["callDup"].year, "2023")
            self.assertEqual(by_call["call123"].content_hash, by_call["callDup"].content_hash)
            self.assertEqual(by_call["call123"].sidecar_metadata["customer.id"], "C1")

            meta = inspect_audio(first)
            self.assertEqual(meta.channels, 2)
            self.assertEqual(meta.sample_rate, 16000)
            self.assertAlmostEqual(meta.duration_sec, 1.0, places=4)

    def test_preprocessing_exports_unmasked_mono_segments_per_channel(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            src = root / "2022" / "call123" / "audio.wav"
            write_stereo_wav(src, seconds=1.2)
            config = PipelineConfig(input_root=root, output_root=Path(td) / "out")
            config.audio.vad_backend = "energy"
            record = discover_audio_files(config)[0]

            segments = preprocess_file(record, config)

            self.assertGreaterEqual(len(segments), 2)
            self.assertEqual({s.channel for s in segments}, {0, 1})
            for segment in segments:
                self.assertTrue(segment.segment_audio_path.exists())
                self.assertEqual(segment.segment_audio_path.parts[-4:-2], ("2022", "segments"))
                sr, channels, rows = read_wav_ints(segment.segment_audio_path)
                self.assertEqual(sr, 16000)
                self.assertEqual(channels, 1)
                self.assertEqual(len(rows), segment.duration_samples)

    def test_decode_to_wav_creates_nested_cache_parent(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "input.opus"
            source.write_bytes(b"fake")
            target = Path(td) / "cache" / "preprocess" / "file1" / "decoded.wav"

            with patch("amc_pipeline.audio.shutil.which", return_value="/usr/bin/ffmpeg"), patch("amc_pipeline.audio.subprocess.run") as run:
                result = decode_to_wav(source, target, sample_rate=16000)

            self.assertEqual(result, target)
            self.assertTrue(target.parent.exists())
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][-1], str(target))

    def test_segmentation_prefers_pause_split_near_target(self):
        intervals = [(0.0, 4.0), (4.6, 9.0), (9.6, 14.0)]

        chunks = split_chunks_on_pauses(
            intervals,
            duration_sec=14.0,
            target_sec=8.0,
            max_sec=10.0,
            min_sec=0.25,
            min_split_gap_sec=0.35,
            merge_gap_sec=0.8,
        )

        self.assertEqual(len(chunks), 2)
        self.assertGreater(chunks[0][1], 9.0)
        self.assertLess(chunks[0][1], 9.6)
        self.assertEqual(chunks[0][1], chunks[1][0])

    def test_redaction_masks_only_target_channel_and_preserves_duration(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.wav"
            out = Path(td) / "redacted.wav"
            write_stereo_wav(src, seconds=1.0)
            before_sr, before_channels, before_rows = read_wav_ints(src)

            result = redact_wav_with_plan(
                src,
                out,
                [MaskInterval(channel=1, start_sec=0.2, end_sec=0.4, reason="test")],
                strategy="silence",
            )

            after_sr, after_channels, after_rows = read_wav_ints(out)
            self.assertTrue(result.ok)
            self.assertEqual(before_sr, after_sr)
            self.assertEqual(before_channels, after_channels)
            self.assertEqual(len(before_rows), len(after_rows))

            start = int(0.2 * before_sr)
            end = int(0.4 * before_sr)
            self.assertEqual([r[0] for r in before_rows], [r[0] for r in after_rows])
            self.assertNotEqual([r[1] for r in before_rows[start:end]], [r[1] for r in after_rows[start:end]])
            self.assertEqual([r[1] for r in before_rows[: start - 20]], [r[1] for r in after_rows[: start - 20]])
            self.assertEqual([r[1] for r in before_rows[end + 20 :]], [r[1] for r in after_rows[end + 20 :]])

    def test_state_upsert_and_pause_are_resume_safe(self):
        with tempfile.TemporaryDirectory() as td:
            state = SQLiteStateStore(Path(td) / "state.db")
            record = {
                "file_id": "f1",
                "source_path": "/tmp/audio.wav",
                "status": "discovered",
                "payload": {"a": 1},
            }
            state.upsert_file(record)
            state.upsert_file({**record, "status": "inspected"})
            state.request_pause(run_id="run1", worker_id="worker1", global_pause=False)

            with closing(sqlite3.connect(Path(td) / "state.db")) as conn:
                file_count = conn.execute("select count(*) from files").fetchone()[0]
            self.assertEqual(file_count, 1)
            self.assertTrue(state.should_pause(run_id="run1", worker_id="worker1"))
            self.assertFalse(state.should_pause(run_id="run1", worker_id="worker2"))

    def test_normalization_consensus_pii_and_alignment(self):
        self.assertEqual(normalize_transcript("User: Call me at FIVE five five 1234! [noise]"), "call me at 555 1234")
        self.assertEqual(normalize_transcript("I'm gonna call twenty five people."), "i am going to call 25 people")

        results = [
            ASRResult(segment_id="s1", model_name="whisper", transcript="John Smith called 555-123-4567", confidence=0.9),
            ASRResult(segment_id="s1", model_name="qwen", transcript="John Smith called 555-123-4567", confidence=0.8),
            ASRResult(segment_id="s1", model_name="cohere", transcript="John Smith called 555-123-4567", confidence=0.7),
            ASRResult(segment_id="s1", model_name="granite", transcript="John called", confidence=0.5),
        ]
        consensus = build_consensus(results, min_successful_models=3)
        self.assertEqual(consensus.method, "strong_exact")
        self.assertEqual(consensus.final_transcript, "John Smith called 555-123-4567")

        pii = RegexPIIDetector().detect("My name is John Smith and phone is 555-123-4567")
        labels = {p.entity_type for p in pii}
        self.assertIn("PHONE", labels)
        self.assertIn("PERSON_NAME", labels)

        transcript = "John Smith called 555-123-4567"
        spans = [PIISpan(entity_type="PHONE", text="555-123-4567", start_char=18, end_char=30, confidence=0.99, source="test")]
        words = TokenUniformAligner().align("s1", transcript, duration_sec=3.0)
        intervals = TokenUniformAligner().spans_to_intervals(spans, transcript, words)
        self.assertEqual(len(intervals), 1)
        self.assertGreater(intervals[0].end_sec, intervals[0].start_sec)

    def test_phone_mask_expands_when_alignment_only_times_prefix(self):
        transcript = "Please call me at 212-537-0830 thank you"
        span = PIISpan("PHONE", "212-537-0830", transcript.index("212"), transcript.index("212") + len("212-537-0830"), 0.99, "regex")
        words = [
            AlignmentWord("Please", 0, 6, 0.0, 0.35),
            AlignmentWord("call", 7, 11, 0.36, 0.62),
            AlignmentWord("me", 12, 14, 0.63, 0.78),
            AlignmentWord("at", 15, 17, 0.79, 0.92),
            AlignmentWord("212", 18, 21, 1.00, 1.35),
            AlignmentWord("thank", 31, 36, 4.20, 4.55),
            AlignmentWord("you", 37, 40, 4.56, 4.78),
        ]

        intervals = TokenUniformAligner().spans_to_intervals([span], transcript, words, channel=1)

        self.assertEqual(intervals[0].channel, 1)
        self.assertGreater(intervals[0].end_sec, 3.5)

    def test_partial_numeric_pii_span_expands_to_full_phone_context(self):
        transcript = "Could you call me at 212-537-0830 today?"
        start = transcript.index("212")
        span = PIISpan("NUMBER", "212", start, start + 3, 0.80, "gliner")

        expanded = _expand_numeric_pii_context(span, transcript)

        self.assertEqual(expanded.text, "212-537-0830")
        self.assertEqual(expanded.entity_type, "PHONE")

    def test_cli_dry_run_writes_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "amc_pipeline.cli",
                    "dry-run",
                    "--input",
                    str(root),
                    "--output",
                    str(out),
                ],
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = out / ".pii_pipeline" / "reports" / "dry_run_summary.json"
            self.assertTrue(summary.exists())
            data = json.loads(summary.read_text())
            self.assertEqual(data["total_files"], 1)

    def test_pipeline_from_seeded_asr_through_redaction_and_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "token_uniform"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)

            preprocess_summary = pipeline.preprocess()
            self.assertEqual(preprocess_summary["segments"], 2)

            transcript = "My name is John Smith and phone is 555-123-4567"
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ["whisper", "qwen", "cohere"]:
                    result = ASRResult(segment_id=segment_id, model_name=model_name, transcript=transcript, confidence=0.9, language="en")
                    pipeline.state.upsert_model_result(segment_id, model_name, "transcribed", dataclass_to_dict(result))

            self.assertEqual(pipeline.normalize()["model_results"], 6)
            self.assertEqual(pipeline.consensus()["strong"], 2)
            self.assertGreaterEqual(pipeline.pii()["spans"], 2)
            self.assertEqual(pipeline.align()["failed"], 0)
            self.assertEqual(pipeline.mask_plan()["files"], 1)
            redacted = pipeline.redact()
            self.assertEqual(len(redacted["outputs"]), 1)
            validation = pipeline.validate()
            self.assertTrue(validation["validations"][0]["ok"])
            self.assertTrue((out / "2022" / "call123" / "audio.wav").exists())

    def test_pipeline_writes_full_output_when_no_pii_is_found(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            source = root / "2022" / "call123" / "audio.wav"
            write_stereo_wav(source)
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "token_uniform"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            transcript = "hello this segment has no private data"
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ["whisper", "qwen", "cohere"]:
                    result = ASRResult(segment_id=segment_id, model_name=model_name, transcript=transcript, confidence=0.9, language="en")
                    pipeline.state.upsert_model_result(segment_id, model_name, "transcribed", dataclass_to_dict(result))

            pipeline.normalize()
            pipeline.consensus()
            self.assertEqual(pipeline.pii()["spans"], 0)
            pipeline.align()
            mask_plan = pipeline.mask_plan()
            self.assertEqual(mask_plan["files"], 1)
            self.assertEqual(mask_plan["intervals"], 0)
            redacted = pipeline.redact()
            self.assertEqual(redacted["failures"], [])
            final_audio = out / "2022" / "call123" / "audio.wav"
            self.assertTrue(final_audio.exists())
            self.assertEqual(source.read_bytes(), final_audio.read_bytes())

    def test_asr_model_failure_is_recorded_without_stopping_other_models(self):
        with tempfile.TemporaryDirectory() as td:
            FailingASRAdapter.closed = False
            StaticASRAdapter.closed = False
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.progress_enabled = False
            pipeline = Pipeline(config)
            pipeline.preprocess()

            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[FailingASRAdapter(), StaticASRAdapter()]):
                summary = pipeline.asr()

            self.assertEqual(summary["models"]["qwen"], 2)
            self.assertEqual(summary["models"]["granite"], 2)
            self.assertEqual(summary["model_failures"][0]["model"], "qwen")
            rows = pipeline.state.fetch_model_results()
            statuses = {(row["model_name"], row["status"]) for row in rows}
            self.assertIn(("qwen", "failed"), statuses)
            self.assertIn(("granite", "transcribed"), statuses)
            self.assertTrue(FailingASRAdapter.closed)
            self.assertTrue(StaticASRAdapter.closed)

    def test_cli_can_override_asr_batch_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            parser = build_parser()
            args = parser.parse_args(
                [
                    "run-stage",
                    "asr",
                    "--input",
                    str(Path(td) / "in"),
                    "--output",
                    str(Path(td) / "out"),
                    "--models",
                    "qwen,cohere,granite",
                    "--asr-batch-sizes",
                    "qwen=1,cohere=2,granite=3",
                ]
            )

            cfg = load_cli_config(args)

            self.assertEqual(cfg.asr_models["qwen"].batch_size, 1)
            self.assertEqual(cfg.asr_models["cohere"].batch_size, 2)
            self.assertEqual(cfg.asr_models["granite"].batch_size, 3)
            self.assertFalse(cfg.asr_models["whisper"].enabled)

    def test_qwen_preflight_requires_thinker_config(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "qwen"
            model_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "config missing"):
                _require_qwen_model_config(model_dir)

            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
            with self.assertRaisesRegex(RuntimeError, "thinker_config"):
                _require_qwen_model_config(model_dir)

            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_asr", "thinker_config": {}}))
            _require_qwen_model_config(model_dir)

    def test_qwen_preflight_rejects_unsupported_transformers_runtime(self):
        with patch("amc_pipeline.transcription.version", return_value="5.9.0"):
            with self.assertRaisesRegex(RuntimeError, "transformers==4.57.6"):
                _require_qwen_runtime_versions()

        with patch("amc_pipeline.transcription.version", return_value="4.57.6"):
            _require_qwen_runtime_versions()

    def test_manifest_stage_includes_transcripts_pii_and_redacted_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "token_uniform"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            transcript = "My name is John Smith"
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ["whisper", "qwen", "cohere"]:
                    result = ASRResult(segment_id=segment_id, model_name=model_name, transcript=transcript, confidence=0.9, language="en")
                    pipeline.state.upsert_model_result(segment_id, model_name, "transcribed", dataclass_to_dict(result))

            pipeline.normalize()
            pipeline.consensus()
            pipeline.pii()
            pipeline.align()
            pipeline.mask_plan()
            pipeline.redact()
            pipeline.validate()
            manifest = pipeline.run_stage("manifest")
            self.assertTrue(manifest["paths"])

            csv_path = out / "manifests" / "all_segments.csv"
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            row = rows[0]
            self.assertIn("whisper_transcript", row)
            self.assertIn("whisper_normalized_transcript", row)
            self.assertIn("final_transcript", row)
            self.assertIn("pii_spans_json", row)
            self.assertIn("mask_intervals_json", row)
            self.assertEqual(row["redacted_audio_path_rel"], "2022/call123/audio.wav")

    def test_pii_can_union_entities_from_all_model_transcripts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.pii_on_all_model_transcripts = True
            config.alignment.backend = "token_uniform"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                pipeline.state.upsert_model_result(segment_id, "whisper", "transcribed", dataclass_to_dict(ASRResult(segment_id, "whisper", "call me at 555-123-4567", 0.9, "en")))
                pipeline.state.upsert_model_result(segment_id, "qwen", "transcribed", dataclass_to_dict(ASRResult(segment_id, "qwen", "call me at 555-123-4567", 0.9, "en")))
                pipeline.state.upsert_model_result(segment_id, "cohere", "transcribed", dataclass_to_dict(ASRResult(segment_id, "cohere", "call me at 555-123-4567", 0.9, "en")))
                pipeline.state.upsert_model_result(segment_id, "granite", "transcribed", dataclass_to_dict(ASRResult(segment_id, "granite", "phone 555-123-4567", 0.9, "en")))

            pipeline.normalize()
            pipeline.consensus()
            pii_summary = pipeline.pii()
            self.assertGreaterEqual(pii_summary["spans"], 2)
            for artifact in pipeline.state.fetch_artifacts("pii"):
                sources = {span["transcript_source"] for span in artifact["payload"]["spans"]}
                self.assertIn("granite", sources)

    def test_run_all_fails_preflight_before_preprocessing_when_models_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            pipeline = Pipeline(config)

            with self.assertRaises(RuntimeError):
                pipeline.run_all()

            self.assertEqual(pipeline.state.fetch_segments(), [])


if __name__ == "__main__":
    unittest.main()
