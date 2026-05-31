from __future__ import annotations

import string
import re
import unicodedata
from functools import lru_cache


_UNITS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_TEENS = {
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
}

_TENS = {
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}

_FILLERS = {
    "uh",
    "um",
    "erm",
    "mm",
    "mmm",
    "mhmm",
    "mhm",
    "hmm",
    "uhhuh",
    "uh-huh",
    "uhuh",
    "ah",
    "eh",
    "aha",
    "huh",
    "hm",
}

_CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "you've": "you have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "let's": "let us",
    "ain't": "is not",
    "y'all": "you all",
    "ma'am": "madam",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
}

_EQUIV_SINGLE = {
    "alright": "all right",
    "allright": "all right",
    "ok": "okay",
    "k": "okay",
    "kay": "okay",
}

_PUNCT_RE = re.compile("[" + re.escape(string.punctuation + "“”’‘") + "]")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", flags=re.IGNORECASE)
_ASR_ARTIFACT_RE = re.compile(
    r"(<\|[^>]*\|>"
    r"|\[(?:music|noise|silence|inaudible|unintelligible|laughs?|coughs?|sighs?|breath(?:ing)?|applause|background[^\]]*)\]"
    r"|\((?:music|noise|silence|inaudible|unintelligible|laughs?|coughs?|sighs?|breath(?:ing)?|applause|background[^\)]*)\))",
    flags=re.IGNORECASE,
)
_PROMPT_ECHO_RE = re.compile(
    r"\b(user|assistant|system)\s*:\s*"
    r"|can you transcribe the speech into a written format\??",
    flags=re.IGNORECASE,
)


def normalize_transcript(text: str, remove_fillers: bool = True) -> str:
    return normalize_transcript_cached(text, remove_fillers=remove_fillers)


@lru_cache(maxsize=500_000)
def normalize_transcript_cached(text: str, remove_fillers: bool = True) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value or value.lower() in {"nan", "<na>", "none"}:
        return ""
    value = cleanup_unicode_lower(value)
    value = _PROMPT_ECHO_RE.sub(" ", value)
    value = _ASR_ARTIFACT_RE.sub(" ", value)
    value = expand_contractions(value)
    value = value.replace("...", " ")
    value = value.replace("-", " ")
    value = _PUNCT_RE.sub(" ", value)
    tokens = re.findall(r"[a-z0-9]+", value)
    if remove_fillers:
        tokens = [t for t in tokens if t not in _FILLERS]
    tokens = normalize_number_tokens(tokens)
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def cleanup_unicode_lower(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    value = value.lower()
    return _ORDINAL_RE.sub(r"\1", value)


def expand_contractions(text: str) -> str:
    value = text
    for key, replacement in _CONTRACTIONS.items():
        value = re.sub(rf"\b{re.escape(key)}\b", replacement, value)
    return value


def normalize_number_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _EQUIV_SINGLE:
            out.extend(_EQUIV_SINGLE[token].split())
            i += 1
            continue
        digit = _unit_digit_or_none(tokens, i)
        if digit is not None:
            j = i
            digits: list[str] = []
            while j < len(tokens):
                next_digit = _unit_digit_or_none(tokens, j)
                if next_digit is None:
                    break
                digits.append(next_digit)
                j += 1
            if len(digits) >= 2:
                out.append("".join(digits))
                i = j
            else:
                out.append(digits[0])
                i += 1
            continue
        if token in _TENS:
            base = int(_TENS[token])
            if i + 1 < len(tokens):
                next_digit = _unit_digit_or_none(tokens, i + 1)
                if next_digit is not None:
                    out.append(str(base + int(next_digit)))
                    i += 2
                    continue
            out.append(str(base))
            i += 1
            continue
        if token in _TEENS:
            out.append(_TEENS[token])
            i += 1
            continue
        if token.isdigit() and len(token) == 1:
            j = i
            digits = []
            while j < len(tokens) and tokens[j].isdigit() and len(tokens[j]) == 1:
                digits.append(tokens[j])
                j += 1
            if len(digits) >= 2:
                out.append("".join(digits))
                i = j
            else:
                out.append(token)
                i += 1
            continue
        out.append(token)
        i += 1
    return out


def _unit_digit_or_none(tokens: list[str], i: int) -> str | None:
    token = tokens[i]
    if token in _UNITS:
        return _UNITS[token]
    if token in {"oh", "o"}:
        prev_num = i > 0 and _is_numeric_context_token(tokens[i - 1])
        next_num = i + 1 < len(tokens) and _is_numeric_context_token(tokens[i + 1])
        if prev_num or next_num:
            return "0"
    return None


def _is_numeric_context_token(token: str) -> bool:
    return token in _UNITS or token in _TEENS or token in _TENS or token in {"oh", "o"} or token.isdigit()


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
