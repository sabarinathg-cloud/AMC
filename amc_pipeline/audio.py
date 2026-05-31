from __future__ import annotations

import math
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from .models import MaskInterval


class MissingDependencyError(RuntimeError):
    pass


@dataclass
class AudioBuffer:
    sample_rate: int
    channels: int
    frames: list[list[float]]

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.sample_rate if self.sample_rate else 0.0


def read_wav(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported by the standard backend: {path}")
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    frames = []
    for idx in range(0, len(values), channels):
        frames.append([max(-1.0, min(1.0, v / 32768.0)) for v in values[idx : idx + channels]])
    return AudioBuffer(sample_rate=sample_rate, channels=channels, frames=frames)


def write_wav(path: Path, buffer: AudioBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    for frame in buffer.frames:
        if len(frame) != buffer.channels:
            raise ValueError("Frame channel count does not match buffer")
        for value in frame:
            clipped = max(-1.0, min(1.0, float(value)))
            payload.extend(struct.pack("<h", int(round(clipped * 32767.0))))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(buffer.channels)
        wf.setsampwidth(2)
        wf.setframerate(buffer.sample_rate)
        wf.writeframes(bytes(payload))


def extract_channel(buffer: AudioBuffer, channel: int) -> list[float]:
    if channel < 0 or channel >= buffer.channels:
        raise ValueError(f"Channel {channel} out of range for {buffer.channels} channels")
    return [frame[channel] for frame in buffer.frames]


def write_mono_segment(path: Path, samples: list[float], sample_rate: int) -> None:
    write_wav(path, AudioBuffer(sample_rate=sample_rate, channels=1, frames=[[x] for x in samples]))


def mask_frames(buffer: AudioBuffer, intervals: list[MaskInterval], strategy: str = "beep", beep_frequency_hz: float = 1000.0, beep_gain: float = 0.35) -> AudioBuffer:
    frames = [list(row) for row in buffer.frames]
    rng = random.Random(1337)
    for interval in intervals:
        if interval.channel < 0 or interval.channel >= buffer.channels:
            raise ValueError(f"Mask channel {interval.channel} out of range")
        start = max(0, int(round(interval.start_sec * buffer.sample_rate)))
        end = min(len(frames), int(round(interval.end_sec * buffer.sample_rate)))
        if end <= start:
            continue
        for i in range(start, end):
            if strategy == "silence":
                replacement = 0.0
            elif strategy == "noise":
                replacement = rng.uniform(-beep_gain, beep_gain)
            elif strategy == "beep":
                replacement = beep_gain * math.sin(2.0 * math.pi * beep_frequency_hz * (i / buffer.sample_rate))
            else:
                raise ValueError(f"Unsupported mask strategy: {strategy}")
            frames[i][interval.channel] = replacement
    return AudioBuffer(sample_rate=buffer.sample_rate, channels=buffer.channels, frames=frames)


def decode_to_wav(source: Path, target_wav: Path, sample_rate: int | None = None) -> Path:
    if source.suffix.lower() == ".wav":
        return source
    if not shutil.which("ffmpeg"):
        raise MissingDependencyError("FFmpeg is required to decode non-WAV audio")
    cmd = ["ffmpeg", "-y", "-i", str(source)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += [str(target_wav)]
    subprocess.run(cmd, check=True, text=True, capture_output=True)
    return target_wav


def encode_from_wav(wav_path: Path, target_path: Path, source_path: Path | None = None) -> Path:
    if target_path.suffix.lower() == ".wav":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(wav_path.read_bytes())
        return target_path
    if not shutil.which("ffmpeg"):
        raise MissingDependencyError("FFmpeg is required to encode non-WAV audio")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(wav_path)]
    if source_path is not None:
        cmd += ["-map_metadata", "0"]
    cmd += [str(target_path)]
    subprocess.run(cmd, check=True, text=True, capture_output=True)
    return target_path


def temp_wav_path(prefix: str = "amc") -> Path:
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_")) / "audio.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

