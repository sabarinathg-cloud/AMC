"""Detect ASR output that repeats itself instead of transcribing.

A streaming/TDT decoder can fall into a loop and emit the same token until the
audio runs out -- in this corpus it happens most often on a phone number read
aloud digit by digit ("four four four four ..."). Two things make that dangerous
rather than merely ugly:

* The loop replaces the exact words that have to be masked, so a segment whose
  transcript is a loop looks PII-free and its audio ships unredacted.
* It is long. One 24s segment produced 9,000 characters, which is enough to make
  a token-classification model allocate 9 GiB of attention and kill the stage.

Consensus cannot filter these on its own: the healthy models write the same
phone number three different ways ("313-524-0075" / "three one three ..."), so
they never form an exact majority and the tie-break falls to the loop -- which
even wins the "longest transcript" tie-break. So we reject the loop up front and
let consensus vote among the models that actually transcribed the audio.

Both signals are deliberately far outside human speech, because a false positive
here silently drops a real transcript:

* speech rate -- fast speech is ~25 chars/sec; the observed loops run 280-370.
* self-repetition -- share of duplicated 5-grams; normal speech sits near 0,
  the observed loops sit at 0.97.
"""
from __future__ import annotations

from dataclasses import dataclass

# Chars/sec above this cannot be speech (fast human speech is ~25).
MAX_CHARS_PER_SEC = 45.0
# The rate check only applies to output long enough for a loop to exist: a 0.3s
# "okay thanks" is a legitimately high rate and must not be flagged.
MIN_CHARS_FOR_RATE_CHECK = 200
# Fraction of repeated n-grams above which the text is looping.
REPEAT_NGRAM = 5
MAX_REPEAT_RATIO = 0.7
MIN_WORDS_FOR_REPEAT_CHECK = 40


@dataclass(frozen=True)
class DegeneracyReport:
    degenerate: bool
    reason: str
    chars_per_sec: float | None
    repeat_ratio: float


def repeat_ratio(text: str, n: int = REPEAT_NGRAM) -> float:
    """Share of n-grams in `text` that are duplicates (0.0 = all distinct)."""
    words = str(text or "").split()
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def inspect_transcript(text: str, duration_sec: float | None = None) -> DegeneracyReport:
    text = str(text or "")
    words = text.split()

    rate: float | None = None
    if duration_sec and duration_sec > 0:
        rate = len(text) / duration_sec
        if len(text) >= MIN_CHARS_FOR_RATE_CHECK and rate > MAX_CHARS_PER_SEC:
            return DegeneracyReport(True, "impossible_speech_rate", rate, repeat_ratio(text))

    ratio = repeat_ratio(text)
    if len(words) >= MIN_WORDS_FOR_REPEAT_CHECK and ratio > MAX_REPEAT_RATIO:
        return DegeneracyReport(True, "repetition_loop", rate, ratio)

    return DegeneracyReport(False, "", rate, ratio)


def is_degenerate(text: str, duration_sec: float | None = None) -> bool:
    return inspect_transcript(text, duration_sec).degenerate
