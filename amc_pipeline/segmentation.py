from __future__ import annotations

import importlib.util
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

try:  # numpy is preferred but optional; the pure-Python path stays fully functional.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised when numpy is absent
    _np = None


_VAD_LOCAL = threading.local()
_VAD_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class PauseCandidate:
    split_time: float
    gap_duration: float
    score: float


def rms(samples) -> float:
    if _np is not None and isinstance(samples, _np.ndarray):
        if samples.size == 0:
            return 0.0
        return float(_np.sqrt(_np.mean(samples.astype(_np.float64) ** 2)))
    if not samples:
        return 0.0
    return math.sqrt(sum(float(x) * float(x) for x in samples) / len(samples))


def energy_vad(
    samples,
    sample_rate: int,
    threshold_ratio: float = 0.08,
    window_ms: int = 30,
    merge_gap_sec: float = 0.8,
    merge: bool = True,
) -> list[tuple[float, float]]:
    n = len(samples)
    if n == 0:
        return []
    window = max(1, int(sample_rate * window_ms / 1000))
    if _np is not None and isinstance(samples, _np.ndarray):
        active_flags = _windowed_active_flags_numpy(samples, window, threshold_ratio)
    else:
        global_rms = rms(samples)
        threshold = max(0.002, global_rms * threshold_ratio)
        active_flags = [rms(samples[start : min(n, start + window)]) >= threshold for start in range(0, n, window)]
    intervals: list[tuple[float, float]] = []
    active_start: int | None = None
    for index, start in enumerate(range(0, n, window)):
        active = active_flags[index]
        if active and active_start is None:
            active_start = start
        elif not active and active_start is not None:
            intervals.append((active_start / sample_rate, start / sample_rate))
            active_start = None
    if active_start is not None:
        intervals.append((active_start / sample_rate, n / sample_rate))
    if merge:
        return merge_intervals(intervals, merge_gap_sec)
    return intervals


def _windowed_active_flags_numpy(samples, window: int, threshold_ratio: float) -> list[bool]:
    arr = samples.astype(_np.float64)
    n = arr.shape[0]
    global_rms = float(_np.sqrt(_np.mean(arr ** 2))) if n else 0.0
    threshold = max(0.002, global_rms * threshold_ratio)
    squared = arr ** 2
    num_full = n // window
    flags: list[bool] = []
    if num_full:
        trimmed = squared[: num_full * window].reshape(num_full, window)
        full_rms = _np.sqrt(trimmed.mean(axis=1))
        flags = [bool(value >= threshold) for value in full_rms]
    if num_full * window < n:
        tail = squared[num_full * window :]
        flags.append(bool(math.sqrt(float(tail.mean())) >= threshold))
    return flags


def vad_intervals(
    samples: list[float],
    sample_rate: int,
    backend: str = "silero",
    threshold_ratio: float = 0.08,
    window_ms: int = 30,
    merge_gap_sec: float = 0.8,
    silero_repo_or_dir: str = "snakers4/silero-vad",
    merge: bool = True,
) -> list[tuple[float, float]]:
    if backend == "silero":
        return silero_vad(samples, sample_rate, merge_gap_sec=merge_gap_sec, repo_or_dir=silero_repo_or_dir, merge=merge)
    if backend == "energy":
        return energy_vad(samples, sample_rate, threshold_ratio=threshold_ratio, window_ms=window_ms, merge_gap_sec=merge_gap_sec, merge=merge)
    raise ValueError(f"Unsupported VAD backend: {backend}")


def preflight_vad_backend(backend: str, silero_repo_or_dir: str = "snakers4/silero-vad") -> None:
    if backend == "energy":
        return
    if backend != "silero":
        raise ValueError(f"Unsupported VAD backend: {backend}")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Silero VAD requires torch")


def _resolve_vad_device() -> str:
    """Pick the Silero VAD device. Defaults to CPU.

    Silero VAD scans audio in ~32 ms windows, i.e. thousands of tiny sequential
    forward passes per call. On GPU each window is a separate micro-kernel whose
    launch+sync overhead dominates, and with N preprocess workers sharing a
    single device those VAD calls serialize. Profiling (ops/profile_preprocess.py)
    showed CPU is faster even single-stream (RTFx ~17 vs ~12), and on CPU the
    per-worker VAD parallelizes across vCPUs instead of bottlenecking on one GPU.
    So CPU is the default; set AMC_VAD_DEVICE=cuda (or cuda:0) to force the GPU.
    """
    override = os.environ.get("AMC_VAD_DEVICE", "").strip()
    if override:
        return override
    return "cpu"


def silero_vad(
    samples: list[float],
    sample_rate: int,
    merge_gap_sec: float = 0.8,
    repo_or_dir: str = "snakers4/silero-vad",
    merge: bool = True,
) -> list[tuple[float, float]]:
    preflight_vad_backend("silero", repo_or_dir)
    import torch  # type: ignore

    model, get_speech_timestamps, device = _get_thread_local_silero(repo_or_dir)
    wav = torch.tensor(samples, dtype=torch.float32)
    if device != "cpu":
        wav = wav.to(device)
    with torch.inference_mode():
        speech_ts = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
    raw = [(float(item["start"] / sample_rate), float(item["end"] / sample_rate)) for item in speech_ts]
    if merge:
        return merge_intervals(raw, merge_gap_sec)
    return raw


def _get_thread_local_silero(repo_or_dir: str):
    cache_key = f"silero_{repo_or_dir}"
    if not hasattr(_VAD_LOCAL, cache_key):
        import torch  # type: ignore

        device = _resolve_vad_device()
        # CPU thread caps only matter for the CPU fallback; on GPU the kernels
        # run on the device so we leave intra-op threading alone.
        if device == "cpu":
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        with _VAD_LOAD_LOCK:
            model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model="silero_vad", trust_repo=True)
        model.eval()
        try:
            model.to(device)
        except Exception:
            # If the GPU placement fails for any reason, fall back to CPU rather
            # than aborting the whole preprocess stage.
            device = "cpu"
            model.to("cpu")
        setattr(_VAD_LOCAL, cache_key, (model, utils[0], device))
    return getattr(_VAD_LOCAL, cache_key)


def merge_intervals(intervals: list[tuple[float, float]], max_gap_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def split_chunks_on_pauses(
    speech_intervals: list[tuple[float, float]],
    duration_sec: float,
    target_sec: float = 12.0,
    max_sec: float = 30.0,
    min_sec: float = 0.25,
    min_split_gap_sec: float = 0.35,
    merge_gap_sec: float = 0.8,
    max_recursion: int = 10,
) -> list[tuple[float, float]]:
    if not speech_intervals:
        return []
    chunks = build_smart_chunks(
        speech_intervals,
        merge_gap_sec=merge_gap_sec,
        target_chunk_sec=target_sec,
        min_keep_chunk_sec=min_sec,
        max_chunk_sec=max_sec,
        min_split_gap_sec=min_split_gap_sec,
        max_recursion=max_recursion,
    )
    out: list[tuple[float, float]] = []
    for start, end in chunks:
        start = max(0.0, min(float(start), duration_sec))
        end = max(0.0, min(float(end), duration_sec))
        if end - start >= min_sec:
            out.append((start, end))
    return out


def build_smart_chunks(
    raw_vad_intervals: list[tuple[float, float]],
    merge_gap_sec: float = 0.8,
    target_chunk_sec: float = 12.0,
    min_keep_chunk_sec: float = 0.25,
    max_chunk_sec: float = 30.0,
    min_split_gap_sec: float = 0.35,
    max_recursion: int = 10,
) -> list[tuple[float, float]]:
    if not raw_vad_intervals:
        return []
    intervals = sorted((float(s), float(e)) for s, e in raw_vad_intervals if e > s)
    if not intervals:
        return []
    grouped_blocks: list[list[tuple[float, float]]] = []
    current_group = [intervals[0]]
    for start, end in intervals[1:]:
        previous_end = current_group[-1][1]
        if start - previous_end <= merge_gap_sec:
            current_group.append((start, end))
        else:
            grouped_blocks.append(current_group)
            current_group = [(start, end)]
    grouped_blocks.append(current_group)

    chunks: list[tuple[float, float]] = []
    for group in grouped_blocks:
        chunks.extend(
            _chunk_long_block(
                block_start=group[0][0],
                block_end=group[-1][1],
                raw_intervals_in_block=group,
                target_chunk_sec=target_chunk_sec,
                min_keep_chunk_sec=min_keep_chunk_sec,
                max_chunk_sec=max_chunk_sec,
                min_split_gap_sec=min_split_gap_sec,
                recursion_depth=0,
                max_recursion=max_recursion,
            )
        )
    return chunks


def find_pause_candidates(
    intervals_in_block: list[tuple[float, float]],
    block_start: float,
    target_chunk_sec: float,
    min_split_gap_sec: float = 0.35,
) -> list[PauseCandidate]:
    candidates: list[PauseCandidate] = []
    for index in range(len(intervals_in_block) - 1):
        left_end = intervals_in_block[index][1]
        right_start = intervals_in_block[index + 1][0]
        gap = right_start - left_end
        if gap < min_split_gap_sec:
            continue
        split_time = (left_end + right_start) / 2.0
        elapsed = split_time - block_start
        distance_penalty = abs(elapsed - target_chunk_sec)
        score = (2.0 * gap) - (0.35 * distance_penalty)
        candidates.append(PauseCandidate(float(split_time), float(gap), float(score)))
    return candidates


def choose_best_split(
    candidates: list[PauseCandidate],
    block_start: float,
    block_end: float,
    min_keep_chunk_sec: float,
    max_chunk_sec: float,
) -> float | None:
    valid: list[tuple[float, PauseCandidate]] = []
    for candidate in candidates:
        left_duration = candidate.split_time - block_start
        right_duration = block_end - candidate.split_time
        if left_duration < min_keep_chunk_sec or right_duration < min_keep_chunk_sec:
            continue
        adjusted_score = candidate.score
        if left_duration > max_chunk_sec:
            adjusted_score -= 5.0
        if right_duration > max_chunk_sec:
            adjusted_score -= 5.0
        valid.append((adjusted_score, candidate))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1].split_time


def _chunk_long_block(
    block_start: float,
    block_end: float,
    raw_intervals_in_block: list[tuple[float, float]],
    target_chunk_sec: float,
    min_keep_chunk_sec: float,
    max_chunk_sec: float,
    min_split_gap_sec: float,
    recursion_depth: int,
    max_recursion: int,
) -> list[tuple[float, float]]:
    duration = block_end - block_start
    if duration < min_keep_chunk_sec:
        return []
    if recursion_depth > max_recursion:
        return [(block_start, block_end)]
    if duration <= max_chunk_sec:
        return [(block_start, block_end)]

    candidates = find_pause_candidates(
        raw_intervals_in_block,
        block_start=block_start,
        target_chunk_sec=target_chunk_sec,
        min_split_gap_sec=min_split_gap_sec,
    )
    split_time = choose_best_split(candidates, block_start, block_end, min_keep_chunk_sec, max_chunk_sec)

    if split_time is None:
        split_time = min(block_start + target_chunk_sec, block_end - min_keep_chunk_sec)
        if split_time <= block_start or split_time >= block_end:
            return [(block_start, block_end)]

    left_intervals = [item for item in raw_intervals_in_block if item[1] <= split_time]
    right_intervals = [item for item in raw_intervals_in_block if item[0] >= split_time]
    if not left_intervals:
        left_intervals = [(block_start, split_time)]
    if not right_intervals:
        right_intervals = [(split_time, block_end)]

    return (
        _chunk_long_block(
            block_start,
            split_time,
            left_intervals,
            target_chunk_sec,
            min_keep_chunk_sec,
            max_chunk_sec,
            min_split_gap_sec,
            recursion_depth + 1,
            max_recursion,
        )
        + _chunk_long_block(
            split_time,
            block_end,
            right_intervals,
            target_chunk_sec,
            min_keep_chunk_sec,
            max_chunk_sec,
            min_split_gap_sec,
            recursion_depth + 1,
            max_recursion,
        )
    )
