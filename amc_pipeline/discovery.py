from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .models import AudioFileRecord


def discover_audio_files(config: PipelineConfig) -> list[AudioFileRecord]:
    if config.input_root is None:
        raise ValueError("input_root is required for discovery")
    root = config.input_root.resolve()
    exts = {e.lower() for e in config.supported_extensions}
    records: list[AudioFileRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.resolve().relative_to(root)
        content_hash = hash_file(path)
        year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        call_id = infer_call_id(path, rel)
        file_id = hashlib.sha256(f"{rel.as_posix()}:{content_hash}".encode("utf-8")).hexdigest()[:24]
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
                duplicate_group_id=content_hash[:24],
                sidecar_metadata=read_sidecar_metadata(path),
            )
        )
    return records


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

