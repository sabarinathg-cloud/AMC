from __future__ import annotations

from collections import Counter

from .models import ASRResult, ConsensusResult
from .normalization import normalize_transcript


# Tie-break order for `weak_best_available`. Only consulted among models that
# actually returned a transcript, so listing every known model here (including
# both whisper and its parakeet replacement) leaves the relative order of an
# existing whisper run unchanged.
DEFAULT_PRIORITY = ["parakeet", "whisper", "qwen", "cohere", "granite"]

# The voting panel: the models whose pairwise agreement defines `model_agreement`.
# Deliberately SEPARATE from DEFAULT_PRIORITY -- the agreement label compares the
# number of models present against the panel size, so folding a new name into the
# tie-break order would otherwise relabel every `all_4_agree` segment in an
# existing run as `incomplete_4_models`. A run overrides this with its own set of
# enabled ASR models; this is only the fallback.
DEFAULT_AGREEMENT_MODELS = ["whisper", "qwen", "cohere", "granite"]


def build_consensus(results: list[ASRResult], min_successful_models: int = 3, priority: list[str] | None = None) -> ConsensusResult:
    priority = priority or DEFAULT_PRIORITY
    ok = [r for r in results if r.ok]
    segment_id = results[0].segment_id if results else ""
    if not ok:
        return ConsensusResult(segment_id, "", "", "failed_no_transcripts", 0.0, None, False)
    normalized = {r.model_name: normalize_transcript(r.transcript, remove_fillers=True) for r in ok}
    counts = Counter(normalized.values())
    best_norm, best_count = counts.most_common(1)[0]
    if best_count >= min_successful_models:
        selected = _choose_for_norm(ok, normalized, best_norm, priority)
        return ConsensusResult(segment_id, selected.transcript, best_norm, "strong_exact", best_count / max(1, len(ok)), selected.model_name, True)
    selected = choose_best_available(ok, priority)
    return ConsensusResult(
        segment_id,
        selected.transcript,
        normalize_transcript(selected.transcript, remove_fillers=True),
        "weak_best_available",
        float(selected.confidence),
        selected.model_name,
        False,
    )


def choose_best_available(results: list[ASRResult], priority: list[str]) -> ASRResult:
    priority_rank = {name: idx for idx, name in enumerate(priority)}
    return sorted(
        results,
        key=lambda r: (
            -(r.confidence or 0.0),
            priority_rank.get(r.model_name, 999),
            -len(r.transcript.split()),
        ),
    )[0]


def _choose_for_norm(results: list[ASRResult], normalized: dict[str, str], target: str, priority: list[str]) -> ASRResult:
    matches = [r for r in results if normalized.get(r.model_name) == target]
    priority_rank = {name: idx for idx, name in enumerate(priority)}
    return sorted(matches, key=lambda r: (priority_rank.get(r.model_name, 999), -(r.confidence or 0.0)))[0]

