from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    data = asdict(obj)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


@dataclass(frozen=True)
class AudioFileRecord:
    file_id: str
    source_path: Path
    relative_path: Path
    year: str
    call_id: str
    extension: str
    size_bytes: int
    content_hash: str
    duplicate_group_id: str
    sidecar_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    codec: str
    container: str
    duration_sec: float
    sample_rate: int
    channels: int
    bitrate: int | None = None
    channel_layout: str | None = None
    format_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    file_id: str
    call_id: str
    year: str
    channel: int
    source_path: Path
    segment_audio_path: Path
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    duration_sec: float
    duration_samples: int
    sample_rate: int
    language: str | None = None
    language_confidence: float | None = None
    status: str = "preprocessed"
    status_reason: str | None = None


@dataclass(frozen=True)
class ASRResult:
    segment_id: str
    model_name: str
    transcript: str
    confidence: float = 0.0
    language: str | None = None
    language_confidence: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.transcript.strip())


@dataclass(frozen=True)
class ConsensusResult:
    segment_id: str
    final_transcript: str
    normalized_transcript: str
    method: str
    score: float
    selected_model: str | None = None
    strong: bool = False


@dataclass(frozen=True)
class PIISpan:
    entity_type: str
    text: str
    start_char: int
    end_char: int
    confidence: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentWord:
    word: str
    start_char: int
    end_char: int
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass(frozen=True)
class MaskInterval:
    channel: int
    start_sec: float
    end_sec: float
    reason: str
    entity_type: str | None = None
    confidence: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
