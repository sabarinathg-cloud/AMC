from __future__ import annotations

from collections import Counter

from .degeneracy import DegeneracyReport, inspect_transcript
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

# `consensus_method` values meaning no model produced a usable transcript, so
# there is no text in which to find PII. Segments landing here are masked whole
# by the align stage rather than shipped unchecked.
UNREADABLE_METHODS = frozenset({"failed_no_transcripts", "failed_all_degenerate"})


def partition_degenerate(
    results: list[ASRResult], duration_sec: float | None = None
) -> tuple[list[ASRResult], list[tuple[ASRResult, DegeneracyReport]]]:
    """Split successful results into (usable, looping). See `degeneracy`.

    A looping transcript must not reach the vote: it is longer than every honest
    transcript, so it wins both the priority and the longest-transcript
    tie-break, and it hides the PII it looped over.
    """
    usable: list[ASRResult] = []
    rejected: list[tuple[ASRResult, DegeneracyReport]] = []
    for result in results:
        if not result.ok:
            continue
        report = inspect_transcript(result.transcript, duration_sec)
        (rejected.append((result, report)) if report.degenerate else usable.append(result))
    return usable, rejected


def build_consensus(
    results: list[ASRResult],
    min_successful_models: int = 3,
    priority: list[str] | None = None,
    duration_sec: float | None = None,
) -> ConsensusResult:
    priority = priority or DEFAULT_PRIORITY
    ok, degenerate = partition_degenerate(results, duration_sec)
    segment_id = results[0].segment_id if results else ""
    if not ok:
        # Every model either errored or looped. There is no transcript to locate
        # PII in, so the segment is reported as having none -- `mask_plan` masks
        # it whole rather than shipping audio nobody has read.
        method = "failed_all_degenerate" if degenerate else "failed_no_transcripts"
        return ConsensusResult(segment_id, "", "", method, 0.0, None, False)
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

