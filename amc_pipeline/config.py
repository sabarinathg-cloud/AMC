from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ROOT = Path("/mnt/amc-data/pipeline/models")


@dataclass
class ASRModelConfig:
    enabled: bool = True
    path: str | None = None
    batch_size: int = 1
    device: str = "auto"


@dataclass
class PIIModelConfig:
    enabled: bool = True
    threshold: float = 0.35
    model_name_or_path: str | None = None


@dataclass
class AudioConfig:
    vad_backend: str = "silero"
    silero_repo_or_dir: str = "snakers4/silero-vad"
    target_sample_rate: int = 16000
    max_segment_sec: float = 30.0
    target_segment_sec: float = 12.0
    min_segment_sec: float = 0.25
    merge_gap_sec: float = 0.8
    min_split_gap_sec: float = 0.35
    max_split_recursion: int = 10
    vad_threshold_ratio: float = 0.08
    vad_window_ms: int = 30


@dataclass
class MaskingConfig:
    strategy: str = "beep"
    pre_padding_ms: int = 80
    post_padding_ms: int = 120
    beep_frequency_hz: float = 1000.0
    beep_gain: float = 0.35
    fade_ms: int = 8
    allow_wav_fallback: bool = True


@dataclass
class StateConfig:
    backend: str = "sqlite"
    postgres_dsn: str | None = None


@dataclass
class AlignmentConfig:
    backend: str = "whisperx"


@dataclass
class PipelineConfig:
    input_root: Path | None = None
    output_root: Path = Path("amc_output")
    supported_extensions: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".opus",
        ".ogg",
        ".flac",
        ".m4a",
        ".aac",
        ".wma",
        ".webm",
    )
    languages: tuple[str, ...] = ("en", "es")
    audio: AudioConfig = field(default_factory=AudioConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    state: StateConfig = field(default_factory=StateConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    asr_models: dict[str, ASRModelConfig] = field(default_factory=dict)
    pii_models: dict[str, PIIModelConfig] = field(default_factory=dict)
    consensus_min_models: int = 3
    pii_on_all_model_transcripts: bool = False
    copy_sidecars: bool = False
    report_mode: str = "internal"
    run_id: str = "default"
    progress_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.asr_models:
            self.asr_models = {
                "whisper": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "whisper-large-v3"), 32),
                "qwen": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "qwen3-asr-1.7b"), 64),
                "cohere": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "cohere-transcribe-03-2026"), 128),
                "granite": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "granite-4.0-1b-speech"), 128),
            }
        if not self.pii_models:
            self.pii_models = {
                "regex": PIIModelConfig(True, 1.0, None),
                "gliner": PIIModelConfig(True, 0.35, "knowledgator/gliner-pii-large-v1.0"),
                "piiranha": PIIModelConfig(True, 0.45, "iiiorg/piiranha-v1-detect-personal-information"),
                "spacy": PIIModelConfig(True, 0.60, "en_core_web_sm"),
                "rule_name": PIIModelConfig(True, 0.99, None),
                "saved_json": PIIModelConfig(True, 1.0, None),
            }
        if self.input_root is not None:
            self.input_root = Path(self.input_root)
        self.output_root = Path(self.output_root)

    @property
    def pipeline_dir(self) -> Path:
        return self.output_root / ".pii_pipeline"

    @property
    def state_db_path(self) -> Path:
        return self.pipeline_dir / "state" / "pipeline.sqlite3"

    @property
    def reports_dir(self) -> Path:
        return self.pipeline_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.pipeline_dir / "cache"

    @property
    def temp_dir(self) -> Path:
        return self.pipeline_dir / "temp"

    def ensure_output_dirs(self) -> None:
        for path in [self.output_root, self.pipeline_dir, self.state_db_path.parent, self.reports_dir, self.cache_dir, self.temp_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_root"] = str(self.input_root) if self.input_root else None
        data["output_root"] = str(self.output_root)
        return data

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: Path, input_root: Path | None = None, output_root: Path | None = None) -> "PipelineConfig":
        raw = _load_mapping(path)
        cfg = cls(
            input_root=Path(raw["input_root"]) if raw.get("input_root") else input_root,
            output_root=Path(raw["output_root"]) if raw.get("output_root") else (output_root or Path("amc_output")),
        )
        _merge_dataclass(cfg.audio, raw.get("audio", {}))
        _merge_dataclass(cfg.masking, raw.get("masking", {}))
        _merge_dataclass(cfg.state, raw.get("state", {}))
        _merge_dataclass(cfg.alignment, raw.get("alignment", {}))
        for name, values in raw.get("asr_models", {}).items():
            current = cfg.asr_models.get(name, ASRModelConfig())
            _merge_dataclass(current, values or {})
            cfg.asr_models[name] = current
        for name, values in raw.get("pii_models", {}).items():
            current = cfg.pii_models.get(name, PIIModelConfig())
            _merge_dataclass(current, values or {})
            cfg.pii_models[name] = current
        for key in ["consensus_min_models", "pii_on_all_model_transcripts", "copy_sidecars", "report_mode", "run_id", "progress_enabled"]:
            if key in raw:
                setattr(cfg, key, raw[key])
        if "languages" in raw:
            cfg.languages = tuple(raw["languages"])
        return cfg


def _merge_dataclass(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
