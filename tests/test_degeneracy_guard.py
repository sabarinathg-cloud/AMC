"""Tests for rejecting looping ASR output.

The compliance case is the important one: a loop replaces the words that have to
be masked, so if it wins the vote the segment looks PII-free and its audio ships
unredacted. See `amc_pipeline.degeneracy`.
"""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from amc_pipeline.config import PipelineConfig
from amc_pipeline.consensus import UNREADABLE_METHODS, build_consensus, partition_degenerate
from amc_pipeline.degeneracy import inspect_transcript, is_degenerate, repeat_ratio
from amc_pipeline.models import ASRResult, dataclass_to_dict
from amc_pipeline.pipeline import Pipeline


def write_stereo_wav(path: Path, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(int(sample_rate * seconds)):
            value = int(3000 * ((i % 100) / 100.0 - 0.5))
            frames += int(value).to_bytes(2, "little", signed=True) * 2
        wf.writeframes(bytes(frames))

# A phone number read digit by digit is where the TDT decoder actually loops.
LOOP = "call us at " + "four " * 600
HONEST = "Hello, this message is for John Martin, please return our call at 313-524-0075."


def result(model: str, transcript: str, confidence: float = 0.0, error: str | None = None) -> ASRResult:
    return ASRResult(
        segment_id="seg-1",
        model_name=model,
        transcript=transcript,
        confidence=confidence,
        language="en",
        language_confidence=0.9,
        error=error,
        raw={},
    )


class DegeneracyDetectorTest(unittest.TestCase):
    def test_flags_a_digit_loop_by_speech_rate(self):
        report = inspect_transcript(LOOP, duration_sec=24.0)
        self.assertTrue(report.degenerate)
        self.assertEqual(report.reason, "impossible_speech_rate")

    def test_flags_a_loop_even_without_duration(self):
        report = inspect_transcript(LOOP, duration_sec=None)
        self.assertTrue(report.degenerate)
        self.assertEqual(report.reason, "repetition_loop")

    def test_keeps_ordinary_speech(self):
        self.assertFalse(is_degenerate(HONEST, 6.0))

    def test_keeps_dense_short_utterances(self):
        # 0.4s of "okay thanks" is a high char rate but obviously not a loop; the
        # rate rule must not fire on output too short to contain one.
        self.assertFalse(is_degenerate("okay thanks", 0.4))

    def test_keeps_a_long_fast_but_varied_transcript(self):
        # ~400 distinct words over two minutes: fast talker, no repetition.
        text = " ".join(f"word{i}" for i in range(400))
        self.assertFalse(is_degenerate(text, 120.0))

    def test_keeps_naturally_repetitive_speech(self):
        # An IVR menu repeats phrases without looping.
        text = ("press one for billing press two for claims press three for pharmacy "
                "press four for eligibility press five to repeat this menu ") * 2
        self.assertFalse(is_degenerate(text, 30.0))

    def test_repeat_ratio_bounds(self):
        self.assertEqual(repeat_ratio("one two three"), 0.0)
        self.assertGreater(repeat_ratio("four " * 100), 0.9)


class ConsensusGuardTest(unittest.TestCase):
    def test_loop_loses_to_the_models_that_transcribed(self):
        # The real shape of the bug: parakeet loops, the healthy models spell the
        # phone number three different ways so none of them form a majority, and
        # the tie-break would otherwise hand the segment to the loop.
        results = [
            result("parakeet", LOOP),
            result("qwen", "please return our call at 313-524-0075"),
            result("cohere", "please return our call at 313 524 0075"),
            result("granite", "please return our call at three one three five two four zero zero seven five"),
        ]
        out = build_consensus(results, min_successful_models=3, duration_sec=24.0)
        self.assertNotEqual(out.selected_model, "parakeet")
        self.assertNotIn("four four four", out.final_transcript)

    def test_agreeing_majority_still_wins(self):
        results = [
            result("parakeet", LOOP),
            result("qwen", HONEST),
            result("cohere", HONEST),
            result("granite", HONEST),
        ]
        out = build_consensus(results, min_successful_models=3, duration_sec=24.0)
        self.assertEqual(out.method, "strong_exact")
        self.assertTrue(out.strong)

    def test_healthy_run_is_untouched(self):
        results = [result(m, HONEST) for m in ("parakeet", "qwen", "cohere", "granite")]
        out = build_consensus(results, min_successful_models=3, duration_sec=6.0)
        self.assertEqual(out.method, "strong_exact")
        self.assertEqual(out.selected_model, "parakeet")

    def test_all_looping_is_reported_as_unreadable(self):
        results = [result(m, LOOP) for m in ("parakeet", "qwen", "cohere", "granite")]
        out = build_consensus(results, min_successful_models=3, duration_sec=24.0)
        self.assertEqual(out.method, "failed_all_degenerate")
        self.assertIn(out.method, UNREADABLE_METHODS)
        self.assertEqual(out.final_transcript, "")

    def test_errored_models_still_report_no_transcripts(self):
        results = [result(m, "", error="boom") for m in ("parakeet", "qwen")]
        out = build_consensus(results, min_successful_models=3, duration_sec=5.0)
        self.assertEqual(out.method, "failed_no_transcripts")
        self.assertIn(out.method, UNREADABLE_METHODS)

    def test_partition_reports_what_it_dropped(self):
        results = [result("parakeet", LOOP), result("qwen", HONEST)]
        usable, rejected = partition_degenerate(results, duration_sec=24.0)
        self.assertEqual([r.model_name for r in usable], ["qwen"])
        self.assertEqual([r.model_name for r, _ in rejected], ["parakeet"])
        self.assertEqual(rejected[0][1].reason, "impossible_speech_rate")


class UnreadableSegmentIsMaskedWholeTest(unittest.TestCase):
    def test_segment_no_model_could_read_is_masked_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root, out = Path(td) / "input", Path(td) / "out"
            write_stereo_wav(root / "2025" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "token_uniform"
            config.progress_enabled = False
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            # Every model loops on this segment, so nothing readable survives the vote.
            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ("parakeet", "qwen", "cohere", "granite"):
                    pipeline.state.upsert_model_result(
                        segment_id, model_name, "transcribed",
                        dataclass_to_dict(result(model_name, LOOP, confidence=0.9)),
                    )

            pipeline.normalize()
            consensus_summary = pipeline.consensus()
            self.assertEqual(consensus_summary["degenerate_rejected"]["parakeet"], 2)

            for artifact in pipeline.state.fetch_artifacts("consensus"):
                self.assertIn(artifact["payload"]["method"], UNREADABLE_METHODS)

            # Regex finds no PII in a loop, so without the fail-safe nothing would be masked.
            self.assertEqual(pipeline.pii()["spans"], 0)
            align_summary = pipeline.align()
            self.assertEqual(align_summary["masked_whole_no_transcript"], 2)

            intervals = [
                interval
                for artifact in pipeline.state.fetch_artifacts("alignment")
                for interval in artifact["payload"]["intervals"]
            ]
            self.assertEqual(len(intervals), 2)
            for interval in intervals:
                self.assertEqual(interval["start_sec"], 0.0)
                self.assertGreater(interval["end_sec"], 0.0)
                self.assertEqual(interval["reason"], "no_usable_transcript")

            mask_plan = pipeline.mask_plan()
            self.assertEqual(mask_plan["intervals"], 2)
            self.assertEqual(pipeline.redact()["failures"], [])
            self.assertTrue((out / "2025" / "call123" / "audio.wav").exists())

    def test_readable_segments_are_not_masked_whole(self):
        with tempfile.TemporaryDirectory() as td:
            root, out = Path(td) / "input", Path(td) / "out"
            write_stereo_wav(root / "2025" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.alignment.backend = "token_uniform"
            config.progress_enabled = False
            for name, detector in config.pii_models.items():
                detector.enabled = name == "regex"
            pipeline = Pipeline(config)
            pipeline.preprocess()

            for segment_row in pipeline.state.fetch_segments():
                segment_id = segment_row["segment_id"]
                for model_name in ("parakeet", "qwen", "cohere"):
                    pipeline.state.upsert_model_result(
                        segment_id, model_name, "transcribed",
                        dataclass_to_dict(result(model_name, HONEST, confidence=0.9)),
                    )

            pipeline.normalize()
            summary = pipeline.consensus()
            self.assertEqual(summary["degenerate_rejected"], {})
            pipeline.pii()
            self.assertEqual(pipeline.align()["masked_whole_no_transcript"], 0)


class PiiranhaWindowingTest(unittest.TestCase):
    """The windowing/batching maths, exercised without loading the model."""

    def _detector(self):
        from amc_pipeline.pii_detection import PiiranhaDetector

        return PiiranhaDetector("unused-path")

    def test_short_text_is_one_window_at_offset_zero(self):
        det = self._detector()
        self.assertEqual(det._windows("hello there"), [(0, "hello there")])

    def test_long_text_is_covered_by_overlapping_windows(self):
        det = self._detector()
        text = " ".join(f"word{i}" for i in range(2000))
        windows = det._windows(text)
        self.assertGreater(len(windows), 1)
        for offset, window in windows:
            self.assertLessEqual(len(window), det.MAX_WINDOW_CHARS)
            self.assertEqual(text[offset : offset + len(window)], window)
        # Consecutive windows overlap, so an entity on a cut is seen whole somewhere.
        for (o1, w1), (o2, _) in zip(windows, windows[1:]):
            self.assertLess(o2, o1 + len(w1))
        self.assertEqual(windows[-1][0] + len(windows[-1][1]), len(text))

    def test_batches_respect_the_char_budget(self):
        det = self._detector()
        windows = [(i, 0, "x" * 1200) for i in range(50)]
        for batch in det._batches(windows):
            longest = max(len(w[2]) for w in batch)
            self.assertLessEqual(len(batch) * longest, det.BATCH_CHAR_BUDGET)
            self.assertLessEqual(len(batch), det.batch_size)

    def test_every_window_is_batched_exactly_once(self):
        det = self._detector()
        windows = [(i, i * 10, "y" * (100 + i * 37)) for i in range(37)]
        seen = [w for batch in det._batches(windows) for w in batch]
        self.assertEqual(sorted(seen), sorted(windows))


if __name__ == "__main__":
    unittest.main()
