from __future__ import annotations

import hashlib
import json
import os
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
    dtype: str = "auto"
    batch_audio_sec_budget: float | None = None
    max_batch_size: int | None = None
    attn_implementation: str | None = None
    prefetch: bool = True
    data_parallel: bool = False


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
class DiscoveryConfig:
    hash_mode: str = "content"
    num_shards: int = 1
    shard_index: int | None = None


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
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    asr_models: dict[str, ASRModelConfig] = field(default_factory=dict)
    pii_models: dict[str, PIIModelConfig] = field(default_factory=dict)
    consensus_min_models: int = 3
    pii_on_all_model_transcripts: bool = False
    copy_sidecars: bool = False
    retain_decoded_cache: bool = False
    preprocess_workers: int = 0
    redact_workers: int = 0
    manifest_dataframe_exports: bool = True
    manifest_xlsx_max_rows: int = 1_000_000
    # At scale a single multi-GB CSV and the per-year duplicates are a write/parse bottleneck
    # over Lustre. Keep them on by default (small runs / back-compat); the orchestrator turns
    # them off for large runs, leaving JSONL (durable) + Parquet (dataset) as the outputs.
    manifest_write_csv: bool = True
    manifest_write_per_year: bool = True
    manifest_parquet_batch_size: int = 50_000
    report_mode: str = "internal"
    run_id: str = "default"
    progress_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.asr_models:
            self.asr_models = {
                # Batch caps from the ops/asr_batch_sweep.py sweep (A10G, caps 8-128). All four
                # max throughput at cap=128 within ~8.3 GB peak; whisper/cohere are byte-identical
                # across caps, qwen/granite show ~1-2.6% benign batching drift absorbed by consensus.
                "whisper": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "whisper-large-v3"), 128),
                "qwen": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "qwen3-asr-1.7b"), 128),
                "cohere": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "cohere-transcribe-03-2026"), 128),
                "granite": ASRModelConfig(True, str(DEFAULT_MODEL_ROOT / "granite-4.0-1b-speech"), 128),
            }
        if not self.pii_models:
            self.pii_models = {
                "regex": PIIModelConfig(True, 1.0, None),
                # gliner disabled by default: the other detectors (regex/piiranha/spacy/
                # rule_name/saved_json) cover PII without it, and it is the slowest model.
                "gliner": PIIModelConfig(False, 0.35, "knowledgator/gliner-pii-large-v1.0"),
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
        # Prefer local scratch (set per-instance by the shard runner) to avoid
        # writing full-length decoded/masked WAVs over shared NFS during redact.
        override = os.environ.get("AMC_TEMP_DIR")
        if override:
            return Path(override)
        return self.pipeline_dir / "temp"

    def ensure_output_dirs(self) -> None:
        for path in [self.output_root, self.pipeline_dir, self.state_db_path.parent, self.reports_dir, self.cache_dir, self.temp_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def resolved_workers(self, requested: int, max_workers: int | None = None) -> int:
        if requested and int(requested) > 0:
            value = int(requested)
        else:
            # Preprocess is dominated by GPU Silero VAD + ffmpeg decode (a
            # subprocess that releases the core), so we oversubscribe the vCPUs
            # to keep the GPU fed and overlap decode with inference. Cap at 8 to
            # avoid thrashing Lustre I/O. Override via preprocess_workers.
            cpu = os.cpu_count() or 1
            value = max(1, min(8, cpu * 2))
        if max_workers is not None:
            value = max(1, min(value, int(max_workers)))
        return value

    @staticmethod
    def available_memory_gb() -> float | None:
        """Best-effort 'usable right now' RAM in GB, or None if undetectable."""
        try:  # Linux: MemAvailable accounts for reclaimable cache, the right figure.
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / (1024.0 * 1024.0)
        except Exception:
            pass
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            return (page * avail) / (1024.0 ** 3)
        except Exception:
            return None

    def resolved_redact_workers(self) -> int:
        """Memory-bounded redact concurrency.

        Redaction decodes a whole call into RAM (float32 per channel) and
        re-encodes it, so concurrency is gated by memory, not CPU. We scale by
        ``MemAvailable / per-call-peak`` so a big-RAM box uses its headroom while
        a 16 GB box stays OOM-safe -- instead of a flat cap of 2. Tunables:
          AMC_REDACT_MEM_PER_CALL_GB  conservative per-call peak (default 2.0)
          AMC_REDACT_MEM_RESERVE_GB   RAM left for the OS/others (default 2.0)
          AMC_REDACT_MAX_WORKERS      absolute ceiling (default 8; I/O-bound past this)
          AMC_REDACT_FORCE=1          let an explicit redact_workers bypass the RAM clamp
        """
        def _envf(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except Exception:
                return default

        per_call_gb = max(0.25, _envf("AMC_REDACT_MEM_PER_CALL_GB", 2.0))
        reserve_gb = max(0.0, _envf("AMC_REDACT_MEM_RESERVE_GB", 2.0))
        hard_cap = max(1, int(_envf("AMC_REDACT_MAX_WORKERS", 8)))
        cpu = os.cpu_count() or 1

        avail = self.available_memory_gb()
        if avail is None:
            ceiling = 2  # unknown memory -> previous safe default
        else:
            usable = max(0.0, avail - reserve_gb)
            ceiling = max(1, int(usable / per_call_gb))
        ceiling = min(ceiling, hard_cap, cpu * 2)

        requested = int(self.redact_workers) if self.redact_workers else 0
        if requested > 0:
            if os.environ.get("AMC_REDACT_FORCE", "0") == "1":
                return requested
            return max(1, min(requested, ceiling))
        return max(1, ceiling)

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
            input_root=input_root if input_root is not None else (Path(raw["input_root"]) if raw.get("input_root") else None),
            output_root=output_root if output_root is not None else (Path(raw["output_root"]) if raw.get("output_root") else Path("amc_output")),
        )
        _merge_dataclass(cfg.audio, raw.get("audio", {}))
        _merge_dataclass(cfg.masking, raw.get("masking", {}))
        _merge_dataclass(cfg.state, raw.get("state", {}))
        _merge_dataclass(cfg.alignment, raw.get("alignment", {}))
        _merge_dataclass(cfg.discovery, raw.get("discovery", {}))
        for name, values in raw.get("asr_models", {}).items():
            current = cfg.asr_models.get(name, ASRModelConfig())
            _merge_dataclass(current, values or {})
            cfg.asr_models[name] = current
        for name, values in raw.get("pii_models", {}).items():
            current = cfg.pii_models.get(name, PIIModelConfig())
            _merge_dataclass(current, values or {})
            cfg.pii_models[name] = current
        for key in ["consensus_min_models", "pii_on_all_model_transcripts", "copy_sidecars", "retain_decoded_cache", "preprocess_workers", "redact_workers", "manifest_dataframe_exports", "manifest_xlsx_max_rows", "manifest_write_csv", "manifest_write_per_year", "manifest_parquet_batch_size", "report_mode", "run_id", "progress_enabled"]:
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
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            if _next_yaml_line_is_child(lines, index, indent):
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = None
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return data


def _next_yaml_line_is_child(lines: list[str], current_index: int, indent: int) -> bool:
    for next_line in lines[current_index + 1 :]:
        if not next_line.strip() or next_line.lstrip().startswith("#"):
            continue
        return len(next_line) - len(next_line.lstrip(" ")) > indent
    return False


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
