from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from .models import AlignmentWord, MaskInterval, PIISpan, SegmentRecord


_DIGIT_WORDS_BY_LANGUAGE = {
    "en": {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    },
    "es": {
        "0": "cero",
        "1": "uno",
        "2": "dos",
        "3": "tres",
        "4": "cuatro",
        "5": "cinco",
        "6": "seis",
        "7": "siete",
        "8": "ocho",
        "9": "nueve",
    },
}
_NUMERIC_SEPARATORS = set(" +-()./#")


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
            start = overlapping[0].start_sec
            end = _conservative_pii_end(span, words, overlapping)
            if _needs_high_recall_numeric_mask(span):
                start = _conservative_pii_start(span, words, overlapping)
            start = max(0.0, start - pre_padding_ms / 1000.0)
            end = end + post_padding_ms / 1000.0
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
        # Cache the per-language forced-alignment model+metadata for the lifetime of this
        # aligner. whisperx.load_align_model() pulls a wav2vec2 model from disk and moves it
        # onto the GPU; doing that once per segment (~240k/shard) made model loading -- not
        # alignment -- the dominant cost (observed ~1.3 seg/s, ~40h/shard). The align stage
        # reuses ONE aligner instance across every segment, so this turns N loads into one
        # per language. Resolve the device once for the same reason.
        self._align_cache: dict[str, tuple] = {}
        self._resolved_device: str | None = None

    def preflight(self) -> None:
        if importlib.util.find_spec("whisperx") is None:
            raise RuntimeError("Forced alignment backend 'whisperx' requires the whisperx package")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Forced alignment backend 'whisperx' requires torch")

    def _device(self) -> str:
        if self._resolved_device is None:
            # Inline import: torch/whisperx exist only in the dedicated `align` venv, so a
            # module-level import would break every other stage that imports this module
            # (e.g. TokenUniformAligner) from the main venv.
            import torch  # type: ignore

            if self.device == "auto":
                self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._resolved_device = self.device
        return self._resolved_device

    def _load_align_model(self, language: str) -> tuple[str, tuple]:
        device = self._device()
        key = f"{device}:{str(language or 'en').lower()}"
        cached = self._align_cache.get(key)
        if cached is None:
            import whisperx  # type: ignore  (align-venv-only dependency; see _device)

            cached = whisperx.load_align_model(language_code=language, device=device)
            self._align_cache[key] = cached
        return device, cached

    def align_segment(self, segment: SegmentRecord, transcript: str, language: str) -> list[AlignmentWord]:
        self.preflight()
        import whisperx  # type: ignore  (align-venv-only dependency; see _device)

        device, (model, metadata) = self._load_align_model(language)
        audio = whisperx.load_audio(str(segment.segment_audio_path))
        alignment_text, source_to_original = _digit_expanded_alignment_text(transcript, language=language)
        aligned = whisperx.align(
            [{"start": 0.0, "end": segment.duration_sec, "text": alignment_text}],
            model,
            metadata,
            audio,
            device,
            return_char_alignments=True,
        )
        units = _alignment_units_from_whisperx(
            transcript,
            aligned,
            alignment_text=alignment_text,
            source_to_original=source_to_original,
        )
        if not units:
            raise RequiredAlignmentError(f"WhisperX returned no usable alignments for {segment.segment_id}")
        return units


def _digit_expanded_alignment_text(transcript: str, language: str | None = None) -> tuple[str, list[int | None]]:
    text = str(transcript or "")
    digit_words = _digit_words_for_language(language)
    out: list[str] = []
    source_to_original: list[int | None] = []
    for idx, char in enumerate(text):
        if char.isdigit():
            _append_alignment_space(out, source_to_original)
            for letter in digit_words[char]:
                out.append(letter)
                source_to_original.append(idx)
            _append_alignment_space(out, source_to_original)
        elif _is_digit_run_separator(text, idx):
            _append_alignment_space(out, source_to_original)
        else:
            out.append(char)
            source_to_original.append(idx)
    return "".join(out).strip(), _trim_alignment_map(out, source_to_original)


def _digit_words_for_language(language: str | None) -> dict[str, str]:
    key = str(language or "en").lower().replace("_", "-").split("-", 1)[0]
    return _DIGIT_WORDS_BY_LANGUAGE.get(key, _DIGIT_WORDS_BY_LANGUAGE["en"])


def _append_alignment_space(out: list[str], source_to_original: list[int | None]) -> None:
    if out and not out[-1].isspace():
        out.append(" ")
        source_to_original.append(None)


def _trim_alignment_map(out: list[str], source_to_original: list[int | None]) -> list[int | None]:
    start = 0
    end = len(out)
    while start < end and out[start].isspace():
        start += 1
    while end > start and out[end - 1].isspace():
        end -= 1
    return source_to_original[start:end]


def _is_digit_run_separator(text: str, idx: int) -> bool:
    char = text[idx]
    if char not in _NUMERIC_SEPARATORS:
        return False
    left = idx - 1
    while left >= 0 and text[left] in _NUMERIC_SEPARATORS:
        left -= 1
    right = idx + 1
    while right < len(text) and text[right] in _NUMERIC_SEPARATORS:
        right += 1
    return left >= 0 and right < len(text) and text[left].isdigit() and text[right].isdigit()


def _alignment_units_from_whisperx(
    transcript: str,
    aligned: dict,
    alignment_text: str | None = None,
    source_to_original: list[int | None] | None = None,
) -> list[AlignmentWord]:
    source_text = alignment_text if alignment_text is not None else transcript
    char_map = source_to_original if source_to_original is not None else list(range(len(source_text)))
    raw_chars = []
    for item in aligned.get("segments", []):
        raw_chars.extend(item.get("chars", []) or [])
    char_units = _chars_with_char_spans(source_text, raw_chars, original_text=transcript, source_to_original=char_map)
    if char_units:
        return char_units
    raw_words = aligned.get("word_segments") or []
    if not raw_words:
        for item in aligned.get("segments", []):
            raw_words.extend(item.get("words", []) or [])
    return _words_with_char_spans(source_text, raw_words, original_text=transcript, source_to_original=char_map)


def _chars_with_char_spans(
    transcript: str,
    raw_chars: list[dict],
    original_text: str | None = None,
    source_to_original: list[int | None] | None = None,
) -> list[AlignmentWord]:
    original_text = transcript if original_text is None else original_text
    source_to_original = source_to_original if source_to_original is not None else list(range(len(transcript)))
    grouped: dict[tuple[int, int], dict[str, float | int | str]] = {}
    order: list[tuple[int, int]] = []
    cursor = 0
    for raw in raw_chars:
        token = str(raw.get("char", raw.get("word", "")))
        if not token or raw.get("start") is None or raw.get("end") is None:
            continue
        found = _find_char_position(transcript, token, cursor)
        if found is None:
            continue
        cursor = found + len(token)
        mapped = _source_span_to_original_span(found, found + len(token), source_to_original)
        if mapped is None:
            continue
        if mapped not in grouped:
            order.append(mapped)
            grouped[mapped] = {
                "word": original_text[mapped[0] : mapped[1]],
                "start_char": mapped[0],
                "end_char": mapped[1],
                "start_sec": float(raw["start"]),
                "end_sec": float(raw["end"]),
                "confidence": float(raw.get("score", 1.0) or 1.0),
            }
        else:
            grouped[mapped]["start_sec"] = min(float(grouped[mapped]["start_sec"]), float(raw["start"]))
            grouped[mapped]["end_sec"] = max(float(grouped[mapped]["end_sec"]), float(raw["end"]))
            grouped[mapped]["confidence"] = max(float(grouped[mapped]["confidence"]), float(raw.get("score", 1.0) or 1.0))
    return [AlignmentWord(**grouped[key]) for key in order]


def _words_with_char_spans(
    transcript: str,
    raw_words: list[dict],
    original_text: str | None = None,
    source_to_original: list[int | None] | None = None,
) -> list[AlignmentWord]:
    original_text = transcript if original_text is None else original_text
    source_to_original = source_to_original if source_to_original is not None else list(range(len(transcript)))
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
        mapped = _source_span_to_original_span(start_char, end_char, source_to_original)
        if mapped is None:
            continue
        start_char, end_char = mapped
        out.append(
            AlignmentWord(
                word=original_text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                start_sec=float(raw["start"]),
                end_sec=float(raw["end"]),
                confidence=float(raw.get("score", 1.0) or 1.0),
            )
        )
    return out


def _source_span_to_original_span(start: int, end: int, source_to_original: list[int | None]) -> tuple[int, int] | None:
    mapped = [source_to_original[idx] for idx in range(max(0, start), min(end, len(source_to_original))) if source_to_original[idx] is not None]
    if not mapped:
        return None
    return min(mapped), max(mapped) + 1


def _find_char_position(transcript: str, token: str, cursor: int) -> int | None:
    if token.isspace():
        match = re.search(re.escape(token), transcript[cursor:])
    else:
        match = re.search(re.escape(token), transcript[cursor:], flags=re.IGNORECASE)
    if match:
        return cursor + match.start()
    if token.strip():
        stripped_cursor = cursor
        while stripped_cursor < len(transcript) and transcript[stripped_cursor].isspace():
            stripped_cursor += 1
        if stripped_cursor < len(transcript) and transcript[stripped_cursor].lower() == token.lower():
            return stripped_cursor
    return None


def _conservative_pii_start(span: PIISpan, words: list[AlignmentWord], overlapping: list[AlignmentWord]) -> float:
    start = overlapping[0].start_sec
    if _span_char_coverage(span, overlapping) >= 0.85:
        return start
    missing_prefix = max(0, overlapping[0].start_char - span.start_char)
    if missing_prefix:
        start = max(0.0, start - _estimated_chars_duration(span.text[:missing_prefix]))
    previous = [w for w in words if w.end_char <= span.start_char and w.end_sec <= overlapping[0].start_sec]
    if previous and missing_prefix:
        start = max(previous[-1].end_sec, start)
    return start


def _conservative_pii_end(span: PIISpan, words: list[AlignmentWord], overlapping: list[AlignmentWord]) -> float:
    end = overlapping[-1].end_sec
    if not _needs_high_recall_numeric_mask(span):
        return end
    coverage = _span_char_coverage(span, overlapping)
    first_start = overlapping[0].start_sec
    current_duration = max(0.0, end - first_start)
    estimated_duration = _estimated_numeric_duration(span.text)
    if coverage >= 0.85 and current_duration >= estimated_duration * 0.65:
        return end
    estimated_end = first_start + estimated_duration
    next_words = [w for w in words if w.start_char >= span.end_char and w.start_sec >= end]
    if next_words:
        estimated_end = max(estimated_end, next_words[0].start_sec)
    return max(end, estimated_end)


def _span_char_coverage(span: PIISpan, words: list[AlignmentWord]) -> float:
    positions = _significant_span_positions(span)
    if not positions:
        positions = set(range(span.start_char, span.end_char))
    covered_positions: set[int] = set()
    for word in words:
        overlap_start = max(span.start_char, word.start_char)
        overlap_end = min(span.end_char, word.end_char)
        covered_positions.update(range(overlap_start, overlap_end))
    return min(1.0, len(positions & covered_positions) / max(1, len(positions)))


def _significant_span_positions(span: PIISpan) -> set[int]:
    value = str(span.text or "")
    if _needs_high_recall_numeric_mask(span):
        digits = {span.start_char + idx for idx, char in enumerate(value) if char.isdigit()}
        if digits:
            return digits
    return {span.start_char + idx for idx, char in enumerate(value) if not char.isspace()}


def _needs_high_recall_numeric_mask(span: PIISpan) -> bool:
    entity = span.entity_type.upper()
    high_risk_entities = {
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
    }
    return entity in high_risk_entities or sum(ch.isdigit() for ch in span.text) >= 4


def _estimated_numeric_duration(text: str) -> float:
    digits = sum(ch.isdigit() for ch in text)
    if digits >= 7:
        return max(2.4, digits * 0.32)
    if digits >= 4:
        return max(1.2, digits * 0.28)
    return _estimated_chars_duration(text)


def _estimated_chars_duration(text: str) -> float:
    return max(0.25, len(str(text or "")) * 0.08)
