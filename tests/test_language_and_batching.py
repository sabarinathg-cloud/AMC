"""Tests for the Parakeet-era language label and the batched state writes.

The language tests cover a compliance path: the align stage picks its forced-alignment
model from the label set here, and aligning Spanish audio with the English model
produces word timings that do not match the words, so the mask intervals derived from
them can miss the PII they were computed for.
"""

import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from amc_pipeline.alignment import WhisperXAligner
from amc_pipeline.config import ASRModelConfig, PipelineConfig
from amc_pipeline.language_id import detect_language
from amc_pipeline.models import ASRResult, SegmentRecord
from amc_pipeline.normalization import normalize_transcript
from amc_pipeline.pipeline import Pipeline
from amc_pipeline.state import SQLiteStateStore
from amc_pipeline.transcription import ParakeetAdapter, language_source_model

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def write_stereo_wav(path: Path, sample_rate: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(nframes):
            value = int(3000 * ((i % 100) / 100.0 - 0.5))
            frames += int(value).to_bytes(2, "little", signed=True) * 2
        wf.writeframes(bytes(frames))


class SpanishThenEnglishAdapter:
    """Stands in for ParakeetAdapter: reports a language it worked out itself."""

    name = "parakeet"
    progress_enabled = False

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        results = []
        for index, segment in enumerate(sorted(segments, key=lambda s: s.segment_id)):
            if index == 0:
                results.append(ASRResult(segment.segment_id, self.name, "Buenas, necesito hablar sobre mi cita", 0.0, "es", 0.9))
            else:
                results.append(ASRResult(segment.segment_id, self.name, "Good morning, this is your nurse calling", 0.0, "en", 0.9))
        return results

    def close(self):
        return None


class LanguageRecordingAdapter:
    """A later panel model; records the language it was handed for each segment."""

    name = "qwen"
    progress_enabled = False

    def __init__(self):
        self.seen: dict[str, str | None] = {}

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        for segment in segments:
            self.seen[segment.segment_id] = segment.language
        return [ASRResult(s.segment_id, self.name, "transcript", 0.5, s.language) for s in segments]

    def close(self):
        return None


class LanguageIdTest(unittest.TestCase):
    def test_spanish_is_detected_from_diacritics_and_function_words(self):
        code, confidence = detect_language("Buenas, este mensaje es para la señora Lucina Román.")
        self.assertEqual(code, "es")
        self.assertGreater(confidence, 0.0)

    def test_spanish_is_detected_without_any_accents(self):
        # The other panel models are inconsistent about emitting accents, so the word
        # lists are matched accent-folded.
        code, _ = detect_language("Buenos dias, necesito hablar con la doctora sobre mi cita de manana")
        self.assertEqual(code, "es")

    def test_english_is_detected(self):
        code, confidence = detect_language("Good morning, this is Tamra, a nurse with AMC Health, calling to follow up.")
        self.assertEqual(code, "en")
        self.assertGreater(confidence, 0.0)

    def test_loanword_heavy_spanish_still_reads_as_spanish(self):
        # Real 2022 transcript: Spanish with an English noun in the middle.
        code, _ = detect_language("un glucómetro en nuestro warehouse del paquete que usted envió")
        self.assertEqual(code, "es")

    def test_no_evidence_returns_none_rather_than_guessing(self):
        for text in ("", "   ", "Mm-hmm.", "Okay."):
            code, confidence = detect_language(text)
            self.assertIsNone(code, f"expected no label for {text!r}")
            self.assertEqual(confidence, 0.0)

    def test_accent_alone_decides_a_short_clip(self):
        # Too short for word evidence, but "í" does not occur in English.
        code, _ = detect_language("Sí.")
        self.assertEqual(code, "es")


class CallLanguageRollupTest(unittest.TestCase):
    """The undecided-segment rollup in ParakeetAdapter.

    Bare digit strings are the motivating case: a spoken phone number has no lexical
    evidence, and it is exactly the PII whose mask interval depends on the align stage
    expanding those digits into the right language's number words.
    """

    @staticmethod
    def _adapter() -> ParakeetAdapter:
        # Construction alone loads no model; that happens in preflight/_load.
        return ParakeetAdapter(ASRModelConfig(path="/nonexistent"))

    @staticmethod
    def _segment(segment_id: str, call_id: str, language: str | None = None) -> SegmentRecord:
        return SegmentRecord(
            segment_id=segment_id,
            file_id="file1",
            call_id=call_id,
            year="2025",
            channel=0,
            source_path=Path("in.wav"),
            segment_audio_path=Path(f"{segment_id}.wav"),
            start_sec=0.0,
            end_sec=1.0,
            start_sample=0,
            end_sample=16000,
            duration_sec=1.0,
            duration_samples=16000,
            sample_rate=16000,
            language=language,
        )

    def test_undecided_segment_inherits_the_rest_of_its_call(self):
        adapter = self._adapter()
        segments = [self._segment("s1", "callA"), self._segment("s2", "callA"), self._segment("s3", "callA")]
        by_id = {
            "s1": ASRResult("s1", "parakeet", "necesito hablar sobre mi cita", 0.0, "es", 1.0),
            "s2": ASRResult("s2", "parakeet", "gracias, hasta luego", 0.0, "es", 1.0),
            "s3": ASRResult("s3", "parakeet", "555 1234", 0.0, None, 0.0),
        }
        adapter._fill_undecided_languages(segments, by_id)
        self.assertEqual(by_id["s3"].language, "es")
        self.assertEqual(by_id["s3"].language_confidence, 1.0)

    def test_confidence_reflects_how_much_of_the_call_agreed(self):
        adapter = self._adapter()
        segments = [self._segment(f"s{i}", "callA") for i in range(1, 5)]
        by_id = {
            "s1": ASRResult("s1", "parakeet", "es", 0.0, "es", 1.0),
            "s2": ASRResult("s2", "parakeet", "es", 0.0, "es", 1.0),
            "s3": ASRResult("s3", "parakeet", "en", 0.0, "en", 1.0),
            "s4": ASRResult("s4", "parakeet", "555 1234", 0.0, None, 0.0),
        }
        adapter._fill_undecided_languages(segments, by_id)
        self.assertEqual(by_id["s4"].language, "es")
        self.assertAlmostEqual(by_id["s4"].language_confidence, 2 / 3)

    def test_a_detected_language_is_never_overridden_by_the_call(self):
        # Mid-call code switching is real in this corpus and must survive the rollup.
        adapter = self._adapter()
        segments = [self._segment("s1", "callA"), self._segment("s2", "callA")]
        by_id = {
            "s1": ASRResult("s1", "parakeet", "necesito hablar sobre mi cita", 0.0, "es", 1.0),
            "s2": ASRResult("s2", "parakeet", "do you speak english, this is your nurse", 0.0, "en", 1.0),
        }
        adapter._fill_undecided_languages(segments, by_id)
        self.assertEqual(by_id["s2"].language, "en")

    def test_calls_do_not_borrow_from_each_other(self):
        adapter = self._adapter()
        segments = [self._segment("s1", "callA"), self._segment("s2", "callB")]
        by_id = {
            "s1": ASRResult("s1", "parakeet", "necesito hablar sobre mi cita", 0.0, "es", 1.0),
            "s2": ASRResult("s2", "parakeet", "555 1234", 0.0, None, 0.0),
        }
        adapter._fill_undecided_languages(segments, by_id)
        self.assertEqual(by_id["s2"].language, "en")
        self.assertEqual(by_id["s2"].language_confidence, 0.0)

    def test_an_entirely_undecidable_call_keeps_the_existing_label(self):
        adapter = self._adapter()
        segments = [self._segment("s1", "callA", language="es"), self._segment("s2", "callA")]
        by_id = {
            "s1": ASRResult("s1", "parakeet", "3549", 0.0, None, 0.0),
            "s2": ASRResult("s2", "parakeet", "", 0.0, None, 0.0),
        }
        adapter._fill_undecided_languages(segments, by_id)
        self.assertEqual(by_id["s1"].language, "es")
        self.assertEqual(by_id["s2"].language, "en")


class LanguageSourceModelTest(unittest.TestCase):
    def test_detecting_model_in_this_run_owns_the_label(self):
        self.assertEqual(language_source_model(["parakeet", "qwen"], []), "parakeet")

    def test_falls_back_to_a_detecting_model_that_already_ran(self):
        # The asr_qwen stage is a separate process: parakeet is not among its adapters,
        # but its rows are already in the DB.
        self.assertEqual(language_source_model(["qwen"], ["parakeet", "qwen"]), "parakeet")

    def test_whisper_is_preferred_when_both_are_present(self):
        self.assertEqual(language_source_model(["parakeet", "whisper"], []), "whisper")
        self.assertEqual(language_source_model(["qwen"], ["parakeet", "whisper"]), "whisper")

    def test_no_detecting_model_anywhere_yields_none(self):
        self.assertIsNone(language_source_model(["qwen", "granite"], ["qwen"]))


class AsrLanguagePropagationTest(unittest.TestCase):
    def _pipeline(self, td: str) -> Pipeline:
        root = Path(td) / "input"
        out = Path(td) / "out"
        write_stereo_wav(root / "2025" / "call123" / "audio.wav")
        config = PipelineConfig(input_root=root, output_root=out)
        config.audio.vad_backend = "energy"
        config.progress_enabled = False
        pipeline = Pipeline(config)
        pipeline.preprocess()
        return pipeline

    def test_parakeet_language_reaches_the_segment_rows(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(td)
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[SpanishThenEnglishAdapter()]):
                pipeline.asr()

            languages = {row["segment_id"]: row["payload"].get("language") for row in pipeline.state.fetch_segments()}
            self.assertEqual(sorted(languages.values()), ["en", "es"])

    def test_later_panel_models_see_the_language_from_a_previous_stage(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(td)
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[SpanishThenEnglishAdapter()]):
                pipeline.asr()

            recorder = LanguageRecordingAdapter()
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[recorder]):
                pipeline.asr()

            self.assertEqual(sorted(recorder.seen.values()), ["en", "es"])


class NormalizeBatchingTest(unittest.TestCase):
    def test_every_row_is_normalized_across_flush_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2025" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.progress_enabled = False
            pipeline = Pipeline(config)
            pipeline.preprocess()
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[SpanishThenEnglishAdapter()]):
                pipeline.asr()

            # Force a flush mid-iteration so the buffered path is the one under test.
            with patch("amc_pipeline.pipeline._NORMALIZE_FLUSH_ROWS", 1):
                first = pipeline.normalize()
            rows = pipeline.state.fetch_model_results()
            self.assertEqual(first["model_results"], len(rows))
            for row in rows:
                payload = row["payload"]
                self.assertEqual(
                    payload["normalized_transcript"],
                    normalize_transcript(payload.get("transcript", ""), remove_fillers=True),
                )

            second = pipeline.normalize()
            self.assertEqual(second["model_results"], 0)
            self.assertEqual(second["skipped_existing"], len(rows))


class RecordArtifactsManyTest(unittest.TestCase):
    def test_batch_write_round_trips_and_updates_on_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            store = SQLiteStateStore(Path(td) / "state.sqlite3")
            store.record_artifacts_many(
                [
                    ("alignment:a", "alignment", Path("a"), "completed", {"segment_id": "a", "words": []}),
                    ("alignment:b", "alignment", Path("b"), "completed", {"segment_id": "b", "words": []}),
                ]
            )
            by_id = {a["artifact_id"]: a for a in store.fetch_artifacts("alignment")}
            self.assertEqual(sorted(by_id), ["alignment:a", "alignment:b"])
            self.assertEqual(by_id["alignment:a"]["payload"]["segment_id"], "a")

            store.record_artifacts_many([("alignment:a", "alignment", Path("a"), "failed", {"segment_id": "a", "error": "boom"})])
            updated = {a["artifact_id"]: a for a in store.fetch_artifacts("alignment")}
            self.assertEqual(len(updated), 2)
            self.assertEqual(updated["alignment:a"]["status"], "failed")

    def test_empty_batch_is_a_noop(self):
        with tempfile.TemporaryDirectory() as td:
            store = SQLiteStateStore(Path(td) / "state.sqlite3")
            store.record_artifacts_many([])
            self.assertEqual(store.fetch_artifacts("alignment"), [])


class AlignEmptySpanBatchingTest(unittest.TestCase):
    def test_segments_without_pii_spans_all_get_an_empty_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_stereo_wav(root / "2025" / "call123" / "audio.wav")
            config = PipelineConfig(input_root=root, output_root=out)
            config.audio.vad_backend = "energy"
            config.progress_enabled = False
            # token_uniform needs no forced-alignment model, so this exercises the stage
            # without the align venv.
            config.alignment.backend = "token_uniform"
            pipeline = Pipeline(config)
            pipeline.preprocess()
            with patch("amc_pipeline.pipeline.build_enabled_adapters", return_value=[SpanishThenEnglishAdapter()]):
                pipeline.asr()
            pipeline.normalize()
            pipeline.consensus()

            segment_ids = [row["segment_id"] for row in pipeline.state.fetch_segments()]
            for segment_id in segment_ids:
                pipeline.state.record_artifact(f"pii:{segment_id}", "pii", Path(segment_id), "completed", {"segment_id": segment_id, "spans": []})

            with patch("amc_pipeline.pipeline._ALIGN_EMPTY_FLUSH_ROWS", 1):
                pipeline.align()

            alignments = {a["payload"]["segment_id"]: a for a in pipeline.state.fetch_artifacts("alignment")}
            self.assertEqual(sorted(alignments), sorted(segment_ids))
            for artifact in alignments.values():
                self.assertEqual(artifact["status"], "completed")
                self.assertEqual(artifact["payload"]["words"], [])
                self.assertEqual(artifact["payload"]["intervals"], [])


@unittest.skipUnless(_HAS_NUMPY, "numpy is only present in the align venv")
class AlignAudioFastPathTest(unittest.TestCase):
    @staticmethod
    def _write_wav(path: Path, channels: int, sample_rate: int, samples: list[int]) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for value in samples:
                frames += int(value).to_bytes(2, "little", signed=True) * channels
            wf.writeframes(bytes(frames))

    def test_matches_the_ffmpeg_scaling_exactly(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seg.wav"
            samples = [0, 1, -1, 32767, -32768, 1234, -4321]
            self._write_wav(path, 1, 16000, samples)
            got = WhisperXAligner._read_pcm16_mono_16k(str(path))
            expected = np.array(samples, dtype=np.int16).astype(np.float32) / 32768.0
            self.assertIsNotNone(got)
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_array_equal(got, expected)

    def test_declines_formats_that_need_real_decoding(self):
        with tempfile.TemporaryDirectory() as td:
            stereo = Path(td) / "stereo.wav"
            self._write_wav(stereo, 2, 16000, [0, 100, -100])
            self.assertIsNone(WhisperXAligner._read_pcm16_mono_16k(str(stereo)))

            resample = Path(td) / "8k.wav"
            self._write_wav(resample, 1, 8000, [0, 100, -100])
            self.assertIsNone(WhisperXAligner._read_pcm16_mono_16k(str(resample)))

            missing = Path(td) / "nope.wav"
            self.assertIsNone(WhisperXAligner._read_pcm16_mono_16k(str(missing)))


if __name__ == "__main__":
    unittest.main()
