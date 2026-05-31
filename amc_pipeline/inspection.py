from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from .models import AudioMetadata


def inspect_audio(path: Path) -> AudioMetadata:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return inspect_wav(path)
    if shutil.which("ffprobe"):
        return inspect_with_ffprobe(path)
    raise RuntimeError(f"FFprobe is required to inspect non-WAV audio: {path}")


def inspect_wav(path: Path) -> AudioMetadata:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
    duration = frames / sample_rate if sample_rate else 0.0
    return AudioMetadata(
        path=path,
        codec=f"pcm_s{sampwidth * 8}le",
        container="wav",
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        bitrate=sample_rate * channels * sampwidth * 8,
        channel_layout="mono" if channels == 1 else "stereo" if channels == 2 else f"{channels} channels",
        format_name="wav",
        raw={"frames": frames, "sample_width": sampwidth},
    )


def inspect_with_ffprobe(path: Path) -> AudioMetadata:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    raw: dict[str, Any] = json.loads(proc.stdout)
    stream = next((s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise RuntimeError(f"No audio stream found: {path}")
    fmt = raw.get("format", {})
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    bitrate_raw = stream.get("bit_rate") or fmt.get("bit_rate")
    return AudioMetadata(
        path=path,
        codec=str(stream.get("codec_name") or "unknown"),
        container=str(fmt.get("format_name") or path.suffix.lstrip(".")),
        duration_sec=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        bitrate=int(bitrate_raw) if bitrate_raw else None,
        channel_layout=stream.get("channel_layout"),
        format_name=fmt.get("format_name"),
        raw=raw,
    )

