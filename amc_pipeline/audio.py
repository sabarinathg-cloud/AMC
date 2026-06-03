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
from typing import Any

from .models import MaskInterval

try:  # numpy is preferred but optional; the pure-Python path stays fully functional.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised when numpy is absent
    _np = None

try:  # soundfile is an optional accelerator for block-streamed reads.
    import soundfile as _sf  # type: ignore
except Exception:  # pragma: no cover - exercised when soundfile is absent
    _sf = None


STREAM_BLOCK_FRAMES = 1 << 18


class MissingDependencyError(RuntimeError):
    pass


@dataclass
class AudioBuffer:
    sample_rate: int
    channels: int
    frames: Any  # list[list[float]] (pure-Python) or np.ndarray of shape (n, channels)

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.sample_rate if self.sample_rate else 0.0


def read_wav(path: Path) -> AudioBuffer:
    if _np is not None:
        return _read_wav_numpy(path)
    return _read_wav_python(path)


def _read_wav_python(path: Path) -> AudioBuffer:
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


def _read_wav_numpy(path: Path) -> AudioBuffer:
    sample_rate, channels, ints = _read_int16(path)
    arr = ints.astype(_np.float64)
    arr /= 32768.0
    _np.clip(arr, -1.0, 1.0, out=arr)
    frames = arr.reshape(-1, channels)
    return AudioBuffer(sample_rate=sample_rate, channels=channels, frames=frames)


def _read_int16(path: Path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported by the standard backend: {path}")
    ints = _np.frombuffer(raw, dtype="<i2")
    return sample_rate, channels, ints


def write_wav(path: Path, buffer: AudioBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _np is not None and isinstance(buffer.frames, _np.ndarray):
        ints = _float_frames_to_int16(buffer.frames, buffer.channels)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(buffer.channels)
            wf.setsampwidth(2)
            wf.setframerate(buffer.sample_rate)
            wf.writeframes(ints.tobytes())
        return
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


def _float_frames_to_int16(frames, channels: int):
    arr = _np.asarray(frames, dtype=_np.float64)
    if arr.ndim == 2 and arr.shape[1] != channels:
        raise ValueError("Frame channel count does not match buffer")
    clipped = _np.clip(arr, -1.0, 1.0)
    return _np.round(clipped * 32767.0).astype("<i2")


def extract_channel(buffer: AudioBuffer, channel: int):
    if channel < 0 or channel >= buffer.channels:
        raise ValueError(f"Channel {channel} out of range for {buffer.channels} channels")
    if _np is not None and isinstance(buffer.frames, _np.ndarray):
        return buffer.frames[:, channel]
    return [frame[channel] for frame in buffer.frames]


def write_mono_segment(path: Path, samples, sample_rate: int) -> None:
    if _np is not None and isinstance(samples, _np.ndarray):
        write_wav(path, AudioBuffer(sample_rate=sample_rate, channels=1, frames=samples.reshape(-1, 1)))
        return
    write_wav(path, AudioBuffer(sample_rate=sample_rate, channels=1, frames=[[x] for x in samples]))


def mask_frames(buffer: AudioBuffer, intervals: list[MaskInterval], strategy: str = "beep", beep_frequency_hz: float = 1000.0, beep_gain: float = 0.35) -> AudioBuffer:
    if _np is not None and isinstance(buffer.frames, _np.ndarray) and strategy != "noise":
        return _mask_frames_numpy(buffer, intervals, strategy, beep_frequency_hz, beep_gain)
    return _mask_frames_python(buffer, intervals, strategy, beep_frequency_hz, beep_gain)


def _mask_frames_python(buffer: AudioBuffer, intervals: list[MaskInterval], strategy: str, beep_frequency_hz: float, beep_gain: float) -> AudioBuffer:
    if _np is not None and isinstance(buffer.frames, _np.ndarray):
        frames = buffer.frames.tolist()
    else:
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
    if _np is not None and isinstance(buffer.frames, _np.ndarray):
        return AudioBuffer(sample_rate=buffer.sample_rate, channels=buffer.channels, frames=_np.asarray(frames, dtype=_np.float64))
    return AudioBuffer(sample_rate=buffer.sample_rate, channels=buffer.channels, frames=frames)


def _mask_frames_numpy(buffer: AudioBuffer, intervals: list[MaskInterval], strategy: str, beep_frequency_hz: float, beep_gain: float) -> AudioBuffer:
    frames = buffer.frames.copy()
    sample_rate = buffer.sample_rate
    n = len(frames)
    for interval in intervals:
        if interval.channel < 0 or interval.channel >= buffer.channels:
            raise ValueError(f"Mask channel {interval.channel} out of range")
        start = max(0, int(round(interval.start_sec * sample_rate)))
        end = min(n, int(round(interval.end_sec * sample_rate)))
        if end <= start:
            continue
        if strategy == "silence":
            frames[start:end, interval.channel] = 0.0
        elif strategy == "beep":
            idx = _np.arange(start, end, dtype=_np.float64)
            frames[start:end, interval.channel] = beep_gain * _np.sin(2.0 * math.pi * beep_frequency_hz * (idx / sample_rate))
        else:
            raise ValueError(f"Unsupported mask strategy: {strategy}")
    return AudioBuffer(sample_rate=sample_rate, channels=buffer.channels, frames=frames)


def redact_wav_streaming(source_path: Path, output_path: Path, intervals: list[MaskInterval], strategy: str, beep_frequency_hz: float = 1000.0, beep_gain: float = 0.35, block_frames: int = STREAM_BLOCK_FRAMES) -> tuple[int, int, int]:
    """Stream a WAV file block-by-block, masking only the requested windows.

    Returns ``(sample_rate, channels, frames_written)``. The full file is never
    materialized; only the active block touches memory. Output is bit-identical
    to ``mask_frames`` + ``write_wav`` for position-deterministic strategies.
    """
    if _np is None:
        raise MissingDependencyError("redact_wav_streaming requires numpy")
    with wave.open(str(source_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported by the standard backend: {source_path}")
    for interval in intervals:
        if interval.channel < 0 or interval.channel >= channels:
            raise ValueError(f"Mask channel {interval.channel} out of range")
    bounds = [
        (
            max(0, int(round(interval.start_sec * sample_rate))),
            int(round(interval.end_sec * sample_rate)),
            interval.channel,
        )
        for interval in intervals
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with wave.open(str(output_path), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(2)
        dst.setframerate(sample_rate)
        block_start = 0
        for int_block in _iter_int16_blocks(source_path, channels, block_frames):
            block = int_block.astype(_np.float64)
            block /= 32768.0
            _np.clip(block, -1.0, 1.0, out=block)
            block_len = block.shape[0]
            block_end = block_start + block_len
            for start, end, channel in bounds:
                clamped_end = min(end, block_end)
                if clamped_end <= start or end <= start:
                    continue
                lo = max(start, block_start)
                hi = min(clamped_end, block_end)
                if hi <= lo:
                    continue
                local_lo = lo - block_start
                local_hi = hi - block_start
                if strategy == "silence":
                    block[local_lo:local_hi, channel] = 0.0
                elif strategy == "beep":
                    idx = _np.arange(lo, hi, dtype=_np.float64)
                    block[local_lo:local_hi, channel] = beep_gain * _np.sin(2.0 * math.pi * beep_frequency_hz * (idx / sample_rate))
                else:
                    raise ValueError(f"Unsupported mask strategy: {strategy}")
            ints = _np.round(_np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2")
            dst.writeframes(ints.tobytes())
            frames_written += block_len
            block_start = block_end
    return sample_rate, channels, frames_written


def _iter_int16_blocks(source_path: Path, channels: int, block_frames: int):
    """Yield contiguous ``(n, channels)`` int16 blocks, preferring soundfile."""
    if _sf is not None:
        for block in _sf.blocks(str(source_path), blocksize=block_frames, dtype="int16", always_2d=True):
            if block.shape[0] == 0:
                continue
            yield _np.ascontiguousarray(block)
        return
    with wave.open(str(source_path), "rb") as src:
        while True:
            raw = src.readframes(block_frames)
            if not raw:
                break
            yield _np.frombuffer(raw, dtype="<i2").reshape(-1, channels)


def wav_frame_count(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getnframes()


def decode_to_wav(source: Path, target_wav: Path, sample_rate: int | None = None) -> Path:
    if source.suffix.lower() == ".wav":
        return source
    if not shutil.which("ffmpeg"):
        raise MissingDependencyError("FFmpeg is required to decode non-WAV audio")
    target_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(source)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += [str(target_wav)]
    _run_ffmpeg(cmd)
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
    _run_ffmpeg(cmd)
    return target_path


def temp_wav_path(prefix: str = "amc", base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(base_dir) if base_dir is not None else None)) / "audio.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        if len(details) > 4000:
            details = details[-4000:]
        joined = " ".join(cmd)
        raise RuntimeError(f"FFmpeg failed with exit code {exc.returncode}: {joined}\n{details}") from exc
