"""Find phone/ID numbers that were SPOKEN digit by digit rather than written.

The neural PII detectors are entirely surface-form dependent. Given the same number
they behave like this:

    "call me at 408-889-9698"                              -> 3 spans
    "call me at four zero eight eight nine nine one six nine eight" -> 0 spans

Whisper hid that blind spot by normalizing numbers into digits, so ~8% of its segments
contained a digit string and the regex/neural detectors caught ~99% of them. Parakeet
transcribes verbatim, which drops digit-form numbers to ~2% of segments and moves the
rest into the word form that nothing detects -- so on 2025 shard-0, 282 of 713 segments
that read out a phone number shipped with no mask at all.

This closes that gap in pure text, with no model and no dependency. Two rules:

* a run of >= MIN_PHONE_RUN spoken digits is a phone number on its own (nobody recites
  seven digits in ordinary conversation), and
* a shorter run is one too when it directly follows a cue like "member id" or
  "date of birth", where a 4-digit answer is the identifier.

Both are deliberately conservative about what counts as a "run": the tokens have to be
CONSECUTIVE, with only punctuation and connector words between them. That is what keeps
ordinary speech out -- "one twenty over eighty, pulse seventy two" breaks at "over" and
"pulse", so it never reaches a maskable length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Spoken digits, including the "oh"/"o" people use for zero.
_DIGIT_WORDS = {
    "zero", "oh", "o", "nought", "naught",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
}
# Numbers above nine still occur inside recited strings ("five five five, twelve twelve",
# "eight hundred"), so they extend a run -- but see _MULTI_ONLY below: they can never
# form a run by themselves, because ordinary speech is full of them.
_MULTI_WORDS = {
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
}
# Spoken shorthand for a repeated digit: "double four", "triple seven".
_REPEAT_WORDS = {"double", "triple"}
# Allowed BETWEEN digits without ending the run: how people punctuate a recited number.
_CONNECTORS = {"and", "dash", "hyphen", "area", "code", "extension", "ext", "plus"}

_TOKEN = re.compile(r"[A-Za-z]+|\d+")

# A run made only of _MULTI_WORDS is ordinary speech ("a hundred and twenty thousand"),
# so a run needs some true single digits before it can be a number at all. An uncued run
# has to carry the whole case by itself and needs several; a cued one needs far fewer,
# because the cue is the evidence -- a spoken date of birth is "one two nineteen fifty",
# only two true digits, and that is exactly the thing we were asked to catch.
_MIN_TRUE_DIGITS = 4
_MIN_CUED_DIGITS = 2

MIN_PHONE_RUN = 7
MIN_CUED_RUN = 4
# How far before a run a cue may sit and still bind to it: "your member ID is four ..."
CUE_WINDOW_CHARS = 40

_CUE = re.compile(
    r"\b(member|account|policy|subscriber|insurance|claim|reference|case|group|"
    r"medicare|medicaid|confirmation|social\s+security|date\s+of\s+birth|dob|ssn|"
    r"phone|number|callback|call\s+back|reached|reach\s+me|extension)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpokenNumberRun:
    start: int
    end: int
    text: str
    entity_type: str
    digits: int


def _tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0).lower()) for m in _TOKEN.finditer(text)]


def _runs(text: str) -> list[tuple[int, int, int, int]]:
    """Consecutive number-word runs as (start_char, end_char, digit_count, token_count)."""
    out: list[tuple[int, int, int, int]] = []
    start = end = None
    digits = tokens = 0
    pending_connector = False

    def flush() -> None:
        nonlocal start, end, digits, tokens, pending_connector
        if start is not None and end is not None and tokens:
            out.append((start, end, digits, tokens))
        start = end = None
        digits = tokens = 0
        pending_connector = False

    for tok_start, tok_end, word in _tokens(text):
        if word in _DIGIT_WORDS or word.isdigit():
            if start is None:
                start = tok_start
            end = tok_end
            digits += 1
            tokens += 1
            pending_connector = False
        elif word in _MULTI_WORDS or word in _REPEAT_WORDS:
            # Extends a run in progress; on its own it is just ordinary speech.
            if start is None:
                continue
            end = tok_end
            tokens += 1
            pending_connector = False
        elif word in _CONNECTORS and start is not None and not pending_connector:
            # Tolerated once between digits; two in a row means the number ended.
            pending_connector = True
        else:
            flush()
    flush()
    return out


def _cue_ends(text: str) -> list[int]:
    return [m.end() for m in _CUE.finditer(text)]


def find_spoken_number_runs(
    text: str,
    min_phone_run: int = MIN_PHONE_RUN,
    min_cued_run: int = MIN_CUED_RUN,
) -> list[SpokenNumberRun]:
    """Spoken phone/ID numbers in `text`, as character spans."""
    text = str(text or "")
    if not text:
        return []
    cues = _cue_ends(text)
    found: list[SpokenNumberRun] = []
    for start, end, digits, tokens in _runs(text):
        cued = any(0 <= start - cue_end <= CUE_WINDOW_CHARS for cue_end in cues)
        if tokens >= min_phone_run and digits >= _MIN_TRUE_DIGITS:
            entity_type = "PHONE"
        elif cued and tokens >= min_cued_run and digits >= _MIN_CUED_DIGITS:
            entity_type = "ID"
        else:
            continue
        found.append(SpokenNumberRun(start, end, text[start:end], entity_type, digits))
    return found
