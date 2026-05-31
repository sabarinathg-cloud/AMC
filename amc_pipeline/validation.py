from __future__ import annotations

from pathlib import Path

from .inspection import inspect_audio
from .models import ValidationResult


def validate_audio_pair(source: Path, output: Path, tolerance_sec: float = 0.010) -> ValidationResult:
    src = inspect_audio(source)
    out = inspect_audio(output)
    duration_ok = abs(src.duration_sec - out.duration_sec) <= tolerance_sec
    channels_ok = src.channels == out.channels
    sample_rate_ok = src.sample_rate == out.sample_rate
    ok = duration_ok and channels_ok and sample_rate_ok
    return ValidationResult(
        ok=ok,
        status="validated" if ok else "failed_validation",
        message="" if ok else "Audio metadata mismatch",
        details={
            "source_duration_sec": src.duration_sec,
            "output_duration_sec": out.duration_sec,
            "source_channels": src.channels,
            "output_channels": out.channels,
            "source_sample_rate": src.sample_rate,
            "output_sample_rate": out.sample_rate,
        },
    )

