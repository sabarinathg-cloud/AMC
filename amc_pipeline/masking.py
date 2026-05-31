from __future__ import annotations

from pathlib import Path

from .audio import mask_frames, read_wav, write_wav
from .models import MaskInterval, ValidationResult


def merge_mask_intervals(intervals: list[MaskInterval], merge_gap_sec: float = 0.02) -> list[MaskInterval]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: (x.channel, x.start_sec, x.end_sec))
    merged: list[MaskInterval] = []
    for interval in sorted_intervals:
        if not merged or interval.channel != merged[-1].channel or interval.start_sec - merged[-1].end_sec > merge_gap_sec:
            merged.append(interval)
            continue
        prev = merged[-1]
        merged[-1] = MaskInterval(
            channel=prev.channel,
            start_sec=min(prev.start_sec, interval.start_sec),
            end_sec=max(prev.end_sec, interval.end_sec),
            reason=f"{prev.reason},{interval.reason}",
            entity_type=prev.entity_type or interval.entity_type,
            confidence=max(prev.confidence or 0.0, interval.confidence or 0.0),
            source=prev.source or interval.source,
        )
    return merged


def redact_wav_with_plan(source_path: Path, output_path: Path, intervals: list[MaskInterval], strategy: str = "beep") -> ValidationResult:
    before = read_wav(source_path)
    clipped = [
        MaskInterval(
            channel=i.channel,
            start_sec=max(0.0, min(before.duration_sec, i.start_sec)),
            end_sec=max(0.0, min(before.duration_sec, i.end_sec)),
            reason=i.reason,
            entity_type=i.entity_type,
            confidence=i.confidence,
            source=i.source,
        )
        for i in intervals
        if i.end_sec > i.start_sec
    ]
    masked = mask_frames(before, merge_mask_intervals(clipped), strategy=strategy)
    write_wav(output_path, masked)
    after = read_wav(output_path)
    ok = before.sample_rate == after.sample_rate and before.channels == after.channels and len(before.frames) == len(after.frames)
    return ValidationResult(
        ok=ok,
        status="completed" if ok else "failed_validation",
        message="" if ok else "Redacted WAV does not match source stream shape",
        details={
            "source_duration_sec": before.duration_sec,
            "output_duration_sec": after.duration_sec,
            "channels": after.channels,
            "sample_rate": after.sample_rate,
            "intervals": len(clipped),
        },
    )

