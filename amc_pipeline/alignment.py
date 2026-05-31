from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from .models import AlignmentWord, MaskInterval, PIISpan, SegmentRecord


class RequiredAlignmentError(RuntimeError):
    pass


class TokenUniformAligner:
    """Deterministic test aligner; production configs should use a local forced aligner."""

    name = "token_uniform"

    def align(self, segment_id: str, transcript: str, duration_sec: float) -> list[AlignmentWord]:
        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", transcript)]
        if not tokens:
            raise RequiredAlignmentError(f"No transcript tokens to align for {segment_id}")
        step = duration_sec / len(tokens)
        words = []
        for idx, (word, start_char, end_char) in enumerate(tokens):
            words.append(AlignmentWord(word, start_char, end_char, idx * step, (idx + 1) * step, 0.5))
        return words

    def spans_to_intervals(
        self,
        spans: list[PIISpan],
        transcript: str,
        words: list[AlignmentWord],
        channel: int = 0,
        pre_padding_ms: int = 80,
        post_padding_ms: int = 120,
    ) -> list[MaskInterval]:
        intervals: list[MaskInterval] = []
        for span in spans:
            overlapping = [w for w in words if not (w.end_char <= span.start_char or w.start_char >= span.end_char)]
            if not overlapping:
                raise RequiredAlignmentError(f"Could not align PII span '{span.text}'")
            start = max(0.0, overlapping[0].start_sec - pre_padding_ms / 1000.0)
            end = overlapping[-1].end_sec + post_padding_ms / 1000.0
            intervals.append(MaskInterval(channel, start, end, span.source, span.entity_type, span.confidence, span.source))
        return intervals


class TorchaudioCTCAligner:
    name = "torchaudio_ctc"

    def preflight(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchaudio") is None:
            raise RuntimeError("Forced alignment requires torch and torchaudio for torchaudio_ctc backend")


class WhisperXAligner:
    name = "whisperx"

    def __init__(self, device: str = "auto"):
        self.device = device

    def preflight(self) -> None:
        if importlib.util.find_spec("whisperx") is None:
            raise RuntimeError("Forced alignment backend 'whisperx' requires the whisperx package")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Forced alignment backend 'whisperx' requires torch")

    def align_segment(self, segment: SegmentRecord, transcript: str, language: str) -> list[AlignmentWord]:
        self.preflight()
        import torch  # type: ignore
        import whisperx  # type: ignore

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else "cpu" if self.device == "auto" else self.device
        audio = whisperx.load_audio(str(segment.segment_audio_path))
        model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(
            [{"start": 0.0, "end": segment.duration_sec, "text": transcript}],
            model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        raw_words = aligned.get("word_segments") or []
        if not raw_words:
            for item in aligned.get("segments", []):
                raw_words.extend(item.get("words", []))
        words = _words_with_char_spans(transcript, raw_words)
        if not words:
            raise RequiredAlignmentError(f"WhisperX returned no word alignments for {segment.segment_id}")
        return words


def _words_with_char_spans(transcript: str, raw_words: list[dict]) -> list[AlignmentWord]:
    out: list[AlignmentWord] = []
    cursor = 0
    for raw in raw_words:
        token = str(raw.get("word", "")).strip()
        if not token or raw.get("start") is None or raw.get("end") is None:
            continue
        match = re.search(re.escape(token), transcript[cursor:], flags=re.IGNORECASE)
        if match:
            start_char = cursor + match.start()
            end_char = cursor + match.end()
            cursor = end_char
        else:
            start_char = cursor
            end_char = min(len(transcript), cursor + len(token))
            cursor = end_char
        out.append(
            AlignmentWord(
                word=token,
                start_char=start_char,
                end_char=end_char,
                start_sec=float(raw["start"]),
                end_sec=float(raw["end"]),
                confidence=float(raw.get("score", 1.0) or 1.0),
            )
        )
    return out
