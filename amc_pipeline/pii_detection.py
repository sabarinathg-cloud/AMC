from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass

from .models import PIISpan
from .config import PipelineConfig


def _pii_device() -> tuple[str, int]:
    """Resolve the device for the neural PII detectors.

    The PII stage runs after ASR, so the GPU is idle and free to use. Returns
    ``(torch_device_str, hf_pipeline_index)`` -- e.g. ("cuda", 0) or ("cpu", -1).
    Override with AMC_PII_DEVICE=cpu|cuda|auto (default auto -> cuda if present).
    """
    pref = os.environ.get("AMC_PII_DEVICE", "auto").strip().lower()
    if pref == "cpu":
        return ("cpu", -1)
    try:
        import torch  # type: ignore

        if pref in ("auto", "cuda", "gpu") and torch.cuda.is_available():
            return ("cuda", 0)
    except Exception:
        pass
    return ("cpu", -1)


class PIIDetector:
    name = "base"

    def preflight(self) -> None:
        return None

    def detect(self, text: str) -> list[PIISpan]:
        raise NotImplementedError

    def detect_batch(self, texts: list[str]) -> list[list[PIISpan]]:
        """Detect PII for many texts at once, returning one span list per input
        (aligned to ``texts``). The default loops ``detect`` so non-batched
        detectors (e.g. regex) keep working; neural detectors override this to
        run a single batched forward pass and saturate the GPU."""
        return [self.detect(t) for t in texts]


class RegexPIIDetector(PIIDetector):
    name = "regex"

    EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    DOB = re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b")
    ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    ID = re.compile(r"\b(?:member|account|policy|reference|case|claim|subscriber|insurance)\s*(?:id|number|no)?\s*[:#-]?\s*([a-z0-9-]{4,})\b", re.IGNORECASE)
    NAME_PATTERNS = [
        re.compile(r"\bmy name is\s+([a-z]+(?:\s+[a-z]+){0,2})\b", re.IGNORECASE),
        re.compile(r"\bthis is\s+([a-z]+(?:\s+[a-z]+){0,2})\b", re.IGNORECASE),
        re.compile(r"\bthis message is for\s+([a-z]+(?:\s+[a-z]+){0,2})\b", re.IGNORECASE),
    ]

    def detect(self, text: str) -> list[PIISpan]:
        text = str(text or "")
        spans: list[PIISpan] = []
        for entity_type, pattern in [
            ("EMAIL", self.EMAIL),
            ("PHONE", self.PHONE),
            ("SSN", self.SSN),
            ("DOB", self.DOB),
            ("ZIP", self.ZIP),
        ]:
            for match in pattern.finditer(text):
                spans.append(PIISpan(entity_type, match.group(0), match.start(), match.end(), 0.99, self.name))
        for match in self.ID.finditer(text):
            start, end = match.span(1)
            spans.append(PIISpan("ID", text[start:end], start, end, 0.95, self.name))
        for pattern in self.NAME_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                name = text[start:end].strip()
                if _likely_name(name):
                    spans.append(PIISpan("PERSON_NAME", name, start, end, 0.99, "rule_name"))
        return resolve_overlaps(spans)


class GLiNERDetector(PIIDetector):
    name = "gliner"

    LABELS = [
        "person name",
        "phone number",
        "email address",
        "location",
        "date",
        "aadhaar number",
        "medical condition",
        "lab result",
        "medication",
        "insurance id",
        "doctor name",
        "institution name",
        "medical code",
    ]

    def __init__(self, model_name_or_path: str, threshold: float = 0.35):
        self.model_name_or_path = model_name_or_path
        self.threshold = threshold
        self._model = None
        self._device = "cpu"
        # How many texts to push through the model per forward pass (GPU memory bound).
        self.batch_size = max(1, int(os.environ.get("AMC_PII_GLINER_BATCH", "32")))

    def preflight(self) -> None:
        if importlib.util.find_spec("gliner") is None:
            raise RuntimeError("GLiNER detector enabled but gliner is not installed")

    def _load(self):
        if self._model is None:
            from gliner import GLiNER  # type: ignore

            self._model = GLiNER.from_pretrained(self.model_name_or_path)
            device, _ = _pii_device()
            if device != "cpu":
                try:
                    self._model = self._model.to(device)
                    self._device = device
                except Exception:
                    self._device = "cpu"
        return self._model

    def _spans_from_raw(self, raw) -> list[PIISpan]:
        return resolve_overlaps(
            [
                PIISpan(str(e["label"]).upper().replace(" ", "_"), str(e["text"]), int(e["start"]), int(e["end"]), float(e.get("score", 0.0)), self.name, dict(e))
                for e in raw
            ]
        )

    def detect(self, text: str) -> list[PIISpan]:
        model = self._load()
        raw = model.predict_entities(str(text), self.LABELS, threshold=self.threshold)
        return self._spans_from_raw(raw)

    def detect_batch(self, texts: list[str]) -> list[list[PIISpan]]:
        model = self._load()
        texts = [str(t or "") for t in texts]
        batch_fn = getattr(model, "batch_predict_entities", None)
        if batch_fn is None:  # older gliner without batching -> per-text fallback
            return [self._spans_from_raw(model.predict_entities(t, self.LABELS, threshold=self.threshold)) for t in texts]
        out: list[list[PIISpan]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            for raw in batch_fn(chunk, self.LABELS, threshold=self.threshold):
                out.append(self._spans_from_raw(raw))
        return out


class PiiranhaDetector(PIIDetector):
    name = "piiranha"

    def __init__(self, model_name_or_path: str, threshold: float = 0.45):
        self.model_name_or_path = model_name_or_path
        self.threshold = threshold
        self._pipe = None
        # Texts per forward pass through the HF pipeline (GPU memory bound).
        self.batch_size = max(1, int(os.environ.get("AMC_PII_PIIRANHA_BATCH", "64")))

    def preflight(self) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("Piiranha detector enabled but transformers is not installed")

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline  # type: ignore

            _, hf_index = _pii_device()
            self._pipe = pipeline(
                "token-classification",
                model=self.model_name_or_path,
                tokenizer=self.model_name_or_path,
                aggregation_strategy="simple",
                device=hf_index,
            )
        return self._pipe

    def _spans_from_items(self, text: str, items) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for item in items:
            score = float(item.get("score", 0.0))
            if score < self.threshold:
                continue
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if start >= 0 and end > start:
                label = str(item.get("entity_group", item.get("entity", "PII"))).upper()
                spans.append(PIISpan(label, str(text)[start:end], start, end, score, self.name, dict(item)))
        return resolve_overlaps(spans)

    def detect(self, text: str) -> list[PIISpan]:
        pipe = self._load()
        text = str(text)
        return self._spans_from_items(text, pipe(text))

    def detect_batch(self, texts: list[str]) -> list[list[PIISpan]]:
        pipe = self._load()
        texts = [str(t or "") for t in texts]
        if not texts:
            return []
        # A list input makes the HF pipeline batch internally (one result list per text).
        results = pipe(texts, batch_size=self.batch_size)
        return [self._spans_from_items(text, items) for text, items in zip(texts, results)]


# spaCy's DATE/CARDINAL entities frequently fire on tokens that are NOT PII: relative time
# references ("today", "tonight"), durations ("two years"), and bare numbers the tagger
# mislabels as dates ("49"). Masking those is pure over-redaction -- it bleeps ordinary speech
# and adds no privacy value -- so we drop low-value temporal/numeric spans. Real PII dates
# (a date of birth, an explicit calendar date, a 4-digit year, a slash/dash date) are kept.
#
# NOTE: the spaCy detector assigns a FLAT confidence (its configured threshold) to every span,
# so raising the threshold cannot single out DATE -- it would drop PERSON/ORG/LOCATION too.
# An allowlist is the correct lever; toggle it off via AMC_MASK_RELATIVE_DATES=1.
_NON_PII_TEMPORAL_WORDS = frozenset({
    "today", "tonight", "tomorrow", "yesterday", "now", "then", "soon", "later",
    "currently", "recently", "lately", "nowadays", "always", "never", "ago",
    "morning", "afternoon", "evening", "night", "midnight", "noon", "midday",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "weekday", "weekdays", "weekend", "weekends",
    "daily", "weekly", "monthly", "quarterly", "yearly", "annually", "annual",
    "hour", "hours", "minute", "minutes", "second", "seconds", "moment", "moments",
    "decade", "decades", "century",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})
# A real calendar date carries a month name or a 4-digit year; if a phrase has neither it is a
# relative reference / duration, not PII.
_MONTHS_RE = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_HAS_MONTH_OR_YEAR_RE = re.compile(rf"\b(?:{_MONTHS_RE})\b|\b\d{{4}}\b", re.IGNORECASE)
# "this/last/next/each/every/few/<n> ..." openers that introduce relative time spans.
_RELATIVE_DATE_RE = re.compile(
    r"^(?:the\s+)?(?:this|that|these|those|last|next|past|coming|recent|each|every|"
    r"few|couple|several|some|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)
_BARE_SMALL_INT_RE = re.compile(r"^\d{1,2}$")


def _is_low_value_temporal(entity_type: str, text: str) -> bool:
    """True for DATE/NUMBER spans that carry no PII (relative time, durations, bare small ints)."""
    if entity_type not in ("DATE", "NUMBER"):
        return False
    norm = (text or "").strip().lower().strip(".,!?;:'\"")
    if not norm:
        return True
    if norm in _NON_PII_TEMPORAL_WORDS:
        return True
    # bare 1-2 digit integers spaCy mislabels as dates ("49", "9"); a real DOB year is 4 digits
    # and slash/dash dates are caught by the regex detector.
    if _BARE_SMALL_INT_RE.match(norm):
        return True
    # relative/duration phrases ("two years", "last week", "a few days") with no month/year.
    if _RELATIVE_DATE_RE.match(norm) and not _HAS_MONTH_OR_YEAR_RE.search(norm):
        return True
    return False


class SpacyDetector(PIIDetector):
    name = "spacy"

    LABEL_MAP = {
        "PERSON": "PERSON_NAME",
        "ORG": "INSTITUTION_NAME",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE",
        "CARDINAL": "NUMBER",
    }

    # NER only needs tok2vec + ner; the tagger/parser/lemmatizer are pure overhead here.
    _NER_ONLY_EXCLUDE = ["tagger", "parser", "lemmatizer", "attribute_ruler", "morphologizer"]

    def __init__(self, model_name_or_path: str = "en_core_web_sm", threshold: float = 0.60):
        self.model_name_or_path = model_name_or_path
        self.threshold = threshold
        self._nlp = None
        # Drop relative/non-PII DATE & NUMBER spans by default; set AMC_MASK_RELATIVE_DATES=1
        # to restore the old behavior of masking every spaCy DATE/CARDINAL hit.
        self.drop_relative_dates = os.environ.get("AMC_MASK_RELATIVE_DATES", "0") != "1"
        self.pipe_batch_size = max(1, int(os.environ.get("AMC_PII_SPACY_BATCH", "256")))

    def preflight(self) -> None:
        if importlib.util.find_spec("spacy") is None:
            raise RuntimeError("spaCy detector enabled but spacy is not installed")

    def _load(self):
        if self._nlp is None:
            import spacy  # type: ignore

            try:
                self._nlp = spacy.load(self.model_name_or_path, exclude=self._NER_ONLY_EXCLUDE)
            except Exception:
                # A component name may not exist in this model; fall back to a full load.
                self._nlp = spacy.load(self.model_name_or_path)
        return self._nlp

    def _spans_from_doc(self, doc) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for ent in doc.ents:
            if ent.label_ not in self.LABEL_MAP:
                continue
            mapped = self.LABEL_MAP[ent.label_]
            if self.drop_relative_dates and _is_low_value_temporal(mapped, ent.text):
                continue
            spans.append(
                PIISpan(
                    mapped,
                    ent.text,
                    int(ent.start_char),
                    int(ent.end_char),
                    self.threshold,
                    self.name,
                    {"spacy_label": ent.label_},
                )
            )
        return resolve_overlaps(spans)

    def detect(self, text: str) -> list[PIISpan]:
        return self._spans_from_doc(self._load()(str(text or "")))

    def detect_batch(self, texts: list[str]) -> list[list[PIISpan]]:
        nlp = self._load()
        docs = nlp.pipe([str(t or "") for t in texts], batch_size=self.pipe_batch_size)
        return [self._spans_from_doc(doc) for doc in docs]


def resolve_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    spans = sorted(spans, key=lambda s: (s.start_char, -(s.end_char - s.start_char), -s.confidence))
    kept: list[PIISpan] = []
    for span in spans:
        overlap_idx = None
        for idx, existing in enumerate(kept):
            if not (span.end_char <= existing.start_char or span.start_char >= existing.end_char):
                overlap_idx = idx
                break
        if overlap_idx is None:
            kept.append(span)
        elif span.confidence > kept[overlap_idx].confidence:
            kept[overlap_idx] = span
    return sorted(kept, key=lambda s: s.start_char)


def _likely_name(name: str) -> bool:
    blacklist = {"doctor", "nurse", "agent", "calling", "message", "phone", "number", "member"}
    toks = [t.lower() for t in name.split()]
    return bool(toks) and len(toks) <= 3 and all(len(t) > 1 and t not in blacklist for t in toks)


def build_enabled_detectors(config: PipelineConfig) -> list[PIIDetector]:
    detectors: list[PIIDetector] = []
    for name, cfg in config.pii_models.items():
        if not cfg.enabled:
            continue
        if name in {"regex", "rule_name"}:
            if not any(isinstance(d, RegexPIIDetector) for d in detectors):
                detectors.append(RegexPIIDetector())
        elif name == "gliner":
            detectors.append(GLiNERDetector(cfg.model_name_or_path or "knowledgator/gliner-pii-large-v1.0", cfg.threshold))
        elif name == "piiranha":
            detectors.append(PiiranhaDetector(cfg.model_name_or_path or "iiiorg/piiranha-v1-detect-personal-information", cfg.threshold))
        elif name == "spacy":
            detectors.append(SpacyDetector(cfg.model_name_or_path or "en_core_web_sm", cfg.threshold))
        elif name == "saved_json":
            continue
        else:
            raise ValueError(f"Unsupported PII detector: {name}")
    return detectors


def preflight_detectors(detectors: list[PIIDetector]) -> None:
    for detector in detectors:
        detector.preflight()
