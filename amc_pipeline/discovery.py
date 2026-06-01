from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from .config import PipelineConfig
from .models import AudioFileRecord


def discover_audio_files(config: PipelineConfig) -> list[AudioFileRecord]:
    if config.input_root is None:
        raise ValueError("input_root is required for discovery")
    _validate_shard_config(config)
    root = config.input_root.resolve()
    exts = {e.lower() for e in config.supported_extensions}
    hash_mode = normalized_hash_mode(config.discovery.hash_mode)
    records: list[AudioFileRecord] = []
    for path in _iter_audio_paths(root, exts):
        rel = path.relative_to(root)
        year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        call_id = infer_call_id(path, rel)
        if not _belongs_to_shard(call_id, config.discovery.num_shards, config.discovery.shard_index):
            continue
        content_hash = fingerprint_file(path, rel, hash_mode)
        file_id = hashlib.sha256(f"{rel.as_posix()}:{content_hash}".encode("utf-8")).hexdigest()[:24]
        duplicate_group_id = content_hash[:24] if hash_mode == "content" else file_id
        records.append(
            AudioFileRecord(
                file_id=file_id,
                source_path=path.resolve(),
                relative_path=rel,
                year=year,
                call_id=call_id,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                content_hash=content_hash,
                duplicate_group_id=duplicate_group_id,
                sidecar_metadata=read_sidecar_metadata(path),
            )
        )
    return records


def _iter_audio_paths(root: Path, exts: set[str]) -> Iterator[Path]:
    seen_dirs: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real_dir = Path(dirpath).resolve()
        if real_dir in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real_dir)
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def infer_call_id(path: Path, relative_path: Path) -> str:
    if path.stem.lower() == "audio" and path.parent.name:
        return path.parent.name
    stable = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{stable}"


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalized_hash_mode(mode: str) -> str:
    return str(mode or "content").lower().replace("-", "_")


def fingerprint_file(path: Path, relative_path: Path, mode: str) -> str:
    mode = normalized_hash_mode(mode)
    if mode == "content":
        return hash_file(path)
    if mode in {"fast", "path"}:
        return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
    if mode in {"path_size_mtime", "metadata"}:
        stat = path.stat()
        payload = f"{relative_path.as_posix()}:{stat.st_size}:{int(stat.st_mtime_ns)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    raise ValueError(f"Unsupported discovery hash mode: {mode}")


def _validate_shard_config(config: PipelineConfig) -> None:
    num_shards = int(config.discovery.num_shards or 1)
    shard_index = config.discovery.shard_index
    config.discovery.num_shards = num_shards
    if num_shards < 1:
        raise ValueError("discovery.num_shards must be >= 1")
    if shard_index is None:
        return
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("discovery.shard_index must be between 0 and num_shards - 1")
    config.discovery.shard_index = shard_index


def _belongs_to_shard(call_id: str, num_shards: int, shard_index: int | None) -> bool:
    if shard_index is None or num_shards <= 1:
        return True
    value = int(hashlib.sha1(str(call_id).encode("utf-8")).hexdigest(), 16)
    return value % num_shards == shard_index


def read_sidecar_metadata(audio_path: Path) -> dict[str, Any]:
    sidecar = audio_path.parent / "metadata.json"
    if not sidecar.exists():
        return {}
    try:
        return flatten_dict(json.loads(sidecar.read_text()))
    except Exception as exc:
        return {"metadata_error": repr(exc)}


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out
