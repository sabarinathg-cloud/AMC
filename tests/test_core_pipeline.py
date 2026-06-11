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

from amc_pipeline.alignment import TokenUniformAligner, _alignment_units_from_whisperx, _digit_expanded_alignment_text
from amc_pipeline.audio import decode_to_wav
from amc_pipeline.cli import build_parser, load_cli_config
from amc_pipeline.config import PipelineConfig, _parse_minimal_yaml
from amc_pipeline.consensus import build_consensus
from amc_pipeline.discovery import discover_audio_files, discovery_counts
from amc_pipeline.inspection import inspect_audio
from amc_pipeline.masking import redact_wav_with_plan
from amc_pipeline.manifests import write_segment_manifests
from amc_pipeline.models import ASRResult, AlignmentWord, MaskInterval, PIISpan, SegmentRecord
from amc_pipeline.normalization import normalize_transcript
from amc_pipeline.pii_detection import RegexPIIDetector, _is_low_value_temporal
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


class CountingWhisperAdapter:
    name = "whisper"
    progress_enabled = False
    calls = 0
    segment_counts = []

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        type(self).calls += 1
        type(self).segment_counts.append(len(segments))
        return [ASRResult(s.segment_id, self.name, "resume safe transcript", 0.9, "en", 0.99) for s in segments]

    def close(self):
        return None


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

    def test_discovery_supports_fast_hash_shards_and_symlinked_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            source_root = Path(td) / "source"
            actual = source_root / "audio.wav"
            write_stereo_wav(actual)
            link = root / "2022" / "call_link" / "audio.wav"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(actual)
            for idx in range(6):
                write_stereo_wav(root / "2022" / f"call_{idx}" / "audio.wav")

            config = PipelineConfig(input_root=root, output_root=Path(td) / "output")
            config.discovery.hash_mode = "path"
            records = discover_audio_files(config)

            by_call = {r.call_id: r for r in records}
            self.assertIn("call_link", by_call)
            self.assertEqual(by_call["call_link"].relative_path.as_posix(), "2022/call_link/audio.wav")
            self.assertEqual(by_call["call_link"].source_path, actual.resolve())

            year_link_root = Path(td) / "year-link-input"
            year_link_root.mkdir()
            (year_link_root / "2022").symlink_to(root / "2022")
            year_link_config = PipelineConfig(input_root=year_link_root, output_root=Path(td) / "year-link-output")
            year_link_config.discovery.hash_mode = "path"
            year_link_records = discover_audio_files(year_link_config)
            self.assertEqual({r.call_id for r in year_link_records}, set(by_call))

            sharded = []
            for shard_index in range(3):
                shard_config = PipelineConfig(input_root=root, output_root=Path(td) / f"out-{shard_index}")
                shard_config.discovery.hash_mode = "path"
                shard_config.discovery.num_shards = "3"  # type: ignore[assignment]
                shard_config.discovery.shard_index = str(shard_index)  # type: ignore[assignment]
                sharded.append({r.call_id for r in discover_audio_files(shard_config)})
            self.assertEqual(set.union(*sharded), set(by_call))
            self.assertFalse(sharded[0] & sharded[1])
            self.assertFalse(sharded[0] & sharded[2])
            self.assertFalse(sharded[1] & sharded[2])

    def test_shard_dirscan_matches_full_walk_per_shard(self):
        # Production points AMC_IN at the year dir, so the layout is <root>/<call_id>/audio.*
        # and AMC_SHARD_DIRSCAN makes each shard enumerate only its own call dirs. The
        # records (and per-shard signature) must be byte-identical to the full-walk+filter
        # path, otherwise existing on-disk stage markers would be invalidated.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "2022"
            for idx in range(20):
                write_stereo_wav(root / f"call_{idx:04d}" / "audio.wav")

            num_shards = 3

            def records_for(shard_index, dirscan):
                cfg = PipelineConfig(input_root=root, output_root=Path(td) / f"o-{dirscan}-{shard_index}")
                cfg.discovery.hash_mode = "path"
                cfg.discovery.num_shards = str(num_shards)  # type: ignore[assignment]
                cfg.discovery.shard_index = str(shard_index)  # type: ignore[assignment]
                env = {"AMC_DISCOVERY_CACHE": "0"}
                env["AMC_SHARD_DIRSCAN"] = "1" if dirscan else "0"
                with patch.dict(os.environ, env, clear=False):
                    recs = discover_audio_files(cfg)
                    counts = discovery_counts(cfg)
                return recs, counts

            union = set()
            for shard_index in range(num_shards):
                walk_recs, walk_counts = records_for(shard_index, dirscan=False)
                scan_recs, scan_counts = records_for(shard_index, dirscan=True)

                def key(records):
                    return sorted(
                        (r.file_id, r.relative_path.as_posix(), r.call_id, r.size_bytes, r.content_hash)
                        for r in records
                    )

                self.assertEqual(key(scan_recs), key(walk_recs))
                # shard file count + signature must match the full-walk path exactly
                self.assertEqual(scan_counts[0], walk_counts[0])
                self.assertEqual(scan_counts[1], walk_counts[1])
                union |= {r.call_id for r in scan_recs}

            # every call lands in exactly one shard
            self.assertEqual(union, {f"call_{idx:04d}" for idx in range(20)})

    def test_input_signature_changes_when_subset_expands(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            output = Path(td) / "output"
            script = Path(__file__).resolve().parents[1] / "ops" / "input_signature.py"
            write_stereo_wav(root / "2026" / "call001" / "audio.wav")

            first = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(root),
                    "--output",
                    str(output),
                    "--hash-mode",
                    "path",
                    "--num-shards",
                    "1",
                    "--shard-index",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            write_stereo_wav(root / "2026" / "call002" / "audio.wav")
            second = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(root),
                    "--output",
                    str(output),
                    "--hash-mode",
                    "path",
                    "--num-shards",
                    "1",
                    "--shard-index",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            first_count, first_signature, first_total = first.stdout.strip().split()
            second_count, second_signature, second_total = second.stdout.strip().split()
            self.assertEqual(first_count, "1")
            self.assertEqual(second_count, "2")
            self.assertEqual(first_total, "1")
            self.assertEqual(second_total, "2")
            self.assertNotEqual(first_signature, second_signature)

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
            state.connect().close()
            state.upsert_file({**record, "status": "reopened"})
            self.assertEqual(state.fetch_files()[0]["status"], "reopened")
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

    def test_phone_mask_expands_when_full_token_time_is_too_short(self):
        transcript = "Could you call me at 212-537-0830 thank you"
        start = transcript.index("212")
        span = PIISpan("PHONE", "212-537-0830", start, start + len("212-537-0830"), 0.99, "regex")
        words = [
            AlignmentWord("Could", 0, 5, 0.0, 0.30),
            AlignmentWord("you", 6, 9, 0.31, 0.48),
            AlignmentWord("call", 10, 14, 0.49, 0.76),
            AlignmentWord("me", 15, 17, 0.77, 0.90),
            AlignmentWord("at", 18, 20, 0.91, 1.02),
            AlignmentWord("212-537-0830", start, start + len("212-537-0830"), 1.10, 1.30),
            AlignmentWord("thank", 34, 39, 4.15, 4.45),
            AlignmentWord("you", 40, 43, 4.46, 4.70),
        ]

        intervals = TokenUniformAligner().spans_to_intervals([span], transcript, words, channel=1)

        self.assertGreater(intervals[0].end_sec, 4.0)

    def test_whisperx_character_alignment_masks_phone_to_final_digit(self):
        transcript = "Could you please give me a call at 212-537-0830? Thank you."
        start = transcript.index("212")
        phone = "212-537-0830"
        digit_times = [9.55, 9.90, 10.22, 10.65, 11.02, 11.38, 11.77, 12.15, 12.55, 13.05]
        chars = []
        phone_digit_idx = 0
        for idx, ch in enumerate(transcript):
            if idx < start:
                chars.append({"char": ch, "start": idx * 0.03, "end": idx * 0.03 + 0.02})
            elif start <= idx < start + len(phone):
                if ch.isdigit():
                    t = digit_times[phone_digit_idx]
                    chars.append({"char": ch, "start": t, "end": t + 0.18})
                    phone_digit_idx += 1
            elif idx > start + len(phone):
                chars.append({"char": ch, "start": 14.0 + idx * 0.02, "end": 14.0 + idx * 0.02 + 0.01})
        aligned = {"segments": [{"chars": chars}]}
        span = PIISpan("PHONE", phone, start, start + len(phone), 0.99, "regex")

        units = _alignment_units_from_whisperx(transcript, aligned)
        intervals = TokenUniformAligner().spans_to_intervals([span], transcript, units, channel=1)

        self.assertLess(intervals[0].start_sec, 9.55)
        self.assertGreater(intervals[0].end_sec, 13.20)
        self.assertLess(intervals[0].end_sec, 13.60)

    def test_digit_expanded_alignment_maps_spoken_phone_digits_to_original_span(self):
        transcript = "Could you please give me a call at 212-537-0830? Thank you."
        start = transcript.index("212")
        phone = "212-537-0830"
        alignment_text, source_to_original = _digit_expanded_alignment_text(transcript)
        digit_positions = [start + idx for idx, char in enumerate(phone) if char.isdigit()]
        digit_times = {pos: 9.55 + idx * 0.42 for idx, pos in enumerate(digit_positions)}
        chars = []
        for pos, char in enumerate(alignment_text):
            orig = source_to_original[pos]
            if orig is None or char.isspace():
                continue
            if orig in digit_times:
                t = digit_times[orig]
                chars.append({"char": char, "start": t, "end": t + 0.18})
            elif orig > start + len(phone):
                t = 14.20 + pos * 0.01
                chars.append({"char": char, "start": t, "end": t + 0.01})
            else:
                t = pos * 0.02
                chars.append({"char": char, "start": t, "end": t + 0.01})
        aligned = {"segments": [{"chars": chars}]}
        span = PIISpan("PHONE", phone, start, start + len(phone), 0.99, "regex")

        units = _alignment_units_from_whisperx(
            transcript,
            aligned,
            alignment_text=alignment_text,
            source_to_original=source_to_original,
        )
        intervals = TokenUniformAligner().spans_to_intervals([span], transcript, units, channel=1)

        self.assertIn("two one two", alignment_text)
        self.assertGreater(intervals[0].end_sec, digit_times[digit_positions[-1]] + 0.20)
        self.assertLess(intervals[0].end_sec, 14.10)

    def test_digit_expanded_alignment_uses_spanish_digit_words(self):
        transcript = "Llame al 212-537-0830 gracias."

        alignment_text, source_to_original = _digit_expanded_alignment_text(transcript, language="es")

        self.assertIn("dos uno dos", alignment_text)
        self.assertNotIn("two one two", alignment_text)
        self.assertEqual(len(alignment_text), len(source_to_original))

    def test_partial_numeric_pii_span_expands_to_full_phone_context(self):
        transcript = "Could you call me at 212-537-0830 today?"
        start = transcript.index("212")
        span = PIISpan("NUMBER", "212", start, start + 3, 0.80, "gliner")

        expanded = _expand_numeric_pii_context(span, transcript)

        self.assertEqual(expanded.text, "212-537-0830")
        self.assertEqual(expanded.entity_type, "PHONE")

    def test_low_value_temporal_dates_are_dropped(self):
        # Relative time refs, durations, and bare small ints carry no PII -> filtered out.
        for text in ["today", "Today.", "tonight", "yesterday", "now", "this week",
                     "last month", "two years", "a few days", "49", "9", "every day"]:
            self.assertTrue(_is_low_value_temporal("DATE", text), f"{text!r} should be dropped")
        self.assertTrue(_is_low_value_temporal("NUMBER", "49"))

    def test_real_dates_are_kept(self):
        # Calendar dates, DOB years, and explicit months are PII -> retained.
        for text in ["March 5th", "1999", "January 1990", "12/05/1987", "March 2020"]:
            self.assertFalse(_is_low_value_temporal("DATE", text), f"{text!r} should be kept")
        # Non temporal entity types are never touched by this filter.
        self.assertFalse(_is_low_value_temporal("PERSON_NAME", "today"))

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

    def test_align_degrades_to_uniform_instead_of_crashing(self):
        from amc_pipeline import pipeline as pipeline_mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "whisperx"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            transcript = "My name is John Smith and phone is 555-123-4567"
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ["whisper", "qwen", "cohere"]:
                    result = ASRResult(segment_id=segment_id, model_name=model_name, transcript=transcript, confidence=0.9, language="en")
                    pipeline.state.upsert_model_result(segment_id, model_name, "transcribed", dataclass_to_dict(result))

            pipeline.normalize()
            pipeline.consensus()
            self.assertGreaterEqual(pipeline.pii()["spans"], 1)

            calls = {"align_segment": 0}

            class _FailingAligner:
                name = "whisperx"

                def __init__(self, *args, **kwargs):
                    pass

                def preflight(self):
                    return None

                def align_segment(self, *args, **kwargs):
                    calls["align_segment"] += 1
                    raise RuntimeError("forced-alignment model load blocked (CVE-2025-32434)")

            original = pipeline_mod.WhisperXAligner
            pipeline_mod.WhisperXAligner = _FailingAligner
            try:
                summary = pipeline.align()
            finally:
                pipeline_mod.WhisperXAligner = original

            # Supported language (en) -> tries the model, fails, degrades to uniform.
            self.assertGreaterEqual(calls["align_segment"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertGreaterEqual(summary["degraded_uniform"], 1)
            # Spans still mask and the call is NOT dropped from output.
            mask_plan = pipeline.mask_plan()
            self.assertEqual(mask_plan["files"], 1)
            self.assertGreaterEqual(mask_plan["intervals"], 1)

    def test_align_skips_language_model_for_unsupported_language(self):
        from amc_pipeline import pipeline as pipeline_mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2022" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "whisperx"
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            transcript = "My name is John Smith and phone is 555-123-4567"
            # Whisper misdetected this segment's language as Portuguese (unsupported).
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ["whisper", "qwen", "cohere"]:
                    result = ASRResult(segment_id=segment_id, model_name=model_name, transcript=transcript, confidence=0.9, language="pt")
                    pipeline.state.upsert_model_result(segment_id, model_name, "transcribed", dataclass_to_dict(result))

            pipeline.normalize()
            pipeline.consensus()
            self.assertGreaterEqual(pipeline.pii()["spans"], 1)

            # The real asr() stage stamps segment.language; here we set it directly to the
            # misdetected unsupported language so the align stage sees it.
            for segment_row in pipeline.state.fetch_segments():
                payload = dict(segment_row["payload"])
                payload["language"] = "pt"
                pipeline.state.upsert_segment(segment_row["segment_id"], segment_row["file_id"], segment_row["status"], payload)

            calls = {"align_segment": 0}

            class _RecordingAligner:
                name = "whisperx"

                def __init__(self, *args, **kwargs):
                    pass

                def preflight(self):
                    return None

                def align_segment(self, *args, **kwargs):
                    calls["align_segment"] += 1
                    raise AssertionError("must not fetch a language model for unsupported language")

            original = pipeline_mod.WhisperXAligner
            pipeline_mod.WhisperXAligner = _RecordingAligner
            try:
                summary = pipeline.align()
            finally:
                pipeline_mod.WhisperXAligner = original

            # Unsupported language must never reach the language-specific model loader.
            self.assertEqual(calls["align_segment"], 0)
            self.assertEqual(summary["failed"], 0)
            self.assertGreaterEqual(summary["degraded_uniform"], 1)
            mask_plan = pipeline.mask_plan()
            self.assertGreaterEqual(mask_plan["intervals"], 1)

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

    def test_stage_resume_skips_completed_items(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2026" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.progress_enabled = False
            pipeline = Pipeline(config)

            first_preprocess = pipeline.preprocess()
            second_preprocess = pipeline.preprocess()
            self.assertEqual(first_preprocess["new_segments"], 2)
            self.assertEqual(second_preprocess["new_segments"], 0)
            self.assertEqual(second_preprocess["skipped_files"], 1)

            CountingWhisperAdapter.calls = 0
            CountingWhisperAdapter.segment_counts = []
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[CountingWhisperAdapter()]):
                first_asr = pipeline.asr()
            self.assertEqual(first_asr["models"]["whisper"], 2)
            self.assertEqual(CountingWhisperAdapter.segment_counts, [2])

            CountingWhisperAdapter.calls = 0
            CountingWhisperAdapter.segment_counts = []
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[CountingWhisperAdapter()]):
                second_asr = pipeline.asr()
            self.assertEqual(second_asr["models"]["whisper"], 0)
            self.assertEqual(second_asr["skipped_existing"]["whisper"], 2)
            self.assertEqual(CountingWhisperAdapter.calls, 0)

            first_normalize = pipeline.normalize()
            second_normalize = pipeline.normalize()
            self.assertEqual(first_normalize["model_results"], 2)
            self.assertEqual(second_normalize["model_results"], 0)
            self.assertEqual(second_normalize["skipped_existing"], 2)

            first_consensus = pipeline.consensus()
            second_consensus = pipeline.consensus()
            self.assertEqual(first_consensus["processed"], 2)
            self.assertEqual(second_consensus["processed"], 0)
            self.assertEqual(second_consensus["skipped_existing"], 2)

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

    def test_cli_can_configure_discovery_sharding(self):
        with tempfile.TemporaryDirectory() as td:
            parser = build_parser()
            args = parser.parse_args(
                [
                    "run",
                    "--input",
                    str(Path(td) / "in"),
                    "--output",
                    str(Path(td) / "out"),
                    "--discovery-hash-mode",
                    "path",
                    "--num-shards",
                    "4",
                    "--shard-index",
                    "2",
                ]
            )

            cfg = load_cli_config(args)

            self.assertEqual(cfg.discovery.hash_mode, "path")
            self.assertEqual(cfg.discovery.num_shards, 4)
            self.assertEqual(cfg.discovery.shard_index, 2)

    def test_cli_paths_override_config_file_paths(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "input_root: /from-config/input",
                        "output_root: /from-config/output",
                        "discovery:",
                        "  hash_mode: content",
                    ]
                )
            )
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "run",
                    "--input",
                    str(Path(td) / "cli-in"),
                    "--output",
                    str(Path(td) / "cli-out"),
                ]
            )

            cfg = load_cli_config(args)

            self.assertEqual(cfg.input_root, Path(td) / "cli-in")
            self.assertEqual(cfg.output_root, Path(td) / "cli-out")

    def test_minimal_yaml_parser_keeps_blank_leaf_values_as_none(self):
        data = _parse_minimal_yaml(
            "\n".join(
                [
                    "discovery:",
                    "  hash_mode: path",
                    "  num_shards: 4",
                    "  shard_index:",
                    "state:",
                    "  backend: sqlite",
                    "  postgres_dsn:",
                ]
            )
        )

        self.assertEqual(data["discovery"]["shard_index"], None)
        self.assertEqual(data["state"]["postgres_dsn"], None)

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

    def _mk_segment(self, i, year="2024"):
        return SegmentRecord(
            segment_id=f"seg{i}", file_id=f"f{i}", call_id=f"c{i}", year=year, channel=i % 2,
            source_path=Path(f"/in/{year}/c/audio.opus"),
            segment_audio_path=Path(f"/out/{year}/segments/c/seg{i}.wav"),
            start_sec=float(i), end_sec=float(i) + 1.0, start_sample=i, end_sample=i + 1,
            duration_sec=1.0, duration_samples=16000, sample_rate=16000, language="en",
        )

    def test_manifest_streaming_unions_columns_and_supports_lean_mode(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            segs = [self._mk_segment(i, "2024" if i % 2 else "2025") for i in range(5)]
            # Only even segments carry PII -> the writer must still emit a stable column union.
            extras = {}
            for i, s in enumerate(segs):
                e = {"whisper_transcript": f"hi {i}", "final_transcript": f"hi {i}"}
                if i % 2 == 0:
                    e["pii_spans_json"] = json.dumps([{"entity_type": "DATE", "text": "today"}])
                    e["pii_count"] = 1
                extras[s.segment_id] = e

            write_segment_manifests(out, iter(segs), extras, parquet_batch_size=2)
            md = out / "manifests"
            with (md / "all_segments.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertIn("pii_spans_json", rows[0])  # union column present even for no-PII rows
            self.assertEqual(len((md / "all_segments.jsonl").read_text().splitlines()), 5)
            self.assertTrue((out / "2024" / "manifests" / "segments.jsonl").exists())
            self.assertFalse(list(md.rglob(".*tmp*")))  # atomic: no temp litter

            lean = Path(td) / "lean"
            lean.mkdir()
            write_segment_manifests(lean, iter(segs), extras, write_csv=False, write_per_year=False)
            self.assertTrue((lean / "manifests" / "all_segments.jsonl").exists())
            self.assertFalse((lean / "manifests" / "all_segments.csv").exists())
            self.assertFalse((lean / "2024").exists())

    def test_manifest_write_is_crash_safe(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            write_segment_manifests(out, [self._mk_segment(99)], {"seg99": {"final_transcript": "keep"}})
            md = out / "manifests"
            before = (md / "all_segments.jsonl").read_text()

            def boom():
                yield self._mk_segment(0)
                raise RuntimeError("simulated crash mid-write")

            with self.assertRaises(RuntimeError):
                write_segment_manifests(out, boom(), {})

            # Prior complete manifest must survive; no partial/corrupt file, no temp litter.
            self.assertEqual((md / "all_segments.jsonl").read_text(), before)
            self.assertIn("keep", before)
            self.assertFalse(list(md.rglob(".*tmp*")))

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
