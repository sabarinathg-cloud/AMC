#!/usr/bin/env python3
"""Export the audio for specific segment IDs so you can LISTEN and decide which
transcript (per-segment baseline vs packed candidate) is actually correct.

The ASR parity check only measures *drift* between two configs, not truth. When
a segment disagrees, the only way to know which side is right is to hear the
clip. This reads the pipeline state DB (same path asr_parity_check uses), then
for each requested SID exports:

  * <sid>.own.{wav,mp3}      -- the segment's own VAD clip == what the
                                per-segment baseline transcribed.
  * <sid>.context.{wav,mp3}  -- a slice of the ORIGINAL source audio covering
                                the segment plus its same-call neighbours, so
                                you can hear whether the packed candidate's
                                extra words actually belong to this segment or
                                leaked in from a neighbour.

mp3s are produced with ffmpeg when available (tiny -- mono 32 kbps). With
--emit-b64 the own-clip mp3 is base64-printed between markers so it can travel
over an SSM stdout (24 KB cap); decode it on your laptop and play.

Example (on the GPU host / via SSM):
    python3 ops/dump_segment_audio.py \
        --input /mnt/amc-data --output /mnt/amc-data/amc-runs/smoke-20/outputs/shard-2 \
        --sid 2026_55d3533c-4105-541a-8e7a-a49dc0702f55_ch0_00019 \
        --context-sec 4 --emit-b64

Then on your Mac, from the printed block:
    pbpaste | sed -n '/-----BEGIN .* OWN MP3-----/,/-----END/p' \
        | grep -v BEGIN | grep -v END | base64 -d > clip.mp3 && afplay clip.mp3
"""
from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import PipelineConfig
from amc_pipeline.pipeline import _segment_from_payload
from amc_pipeline.state import PostgresStateStore, SQLiteStateStore


def _build_state_store(config: PipelineConfig):
    config.ensure_output_dirs()
    if config.state.backend == "sqlite":
        return SQLiteStateStore(config.state_db_path)
    if config.state.backend == "postgres":
        return PostgresStateStore(config.state.postgres_dsn or "")
    raise RuntimeError(f"Unsupported state backend: {config.state.backend}")


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        return PipelineConfig.from_file(
            Path(args.config),
            input_root=Path(args.input) if args.input else None,
            output_root=Path(args.output) if args.output else None,
        )
    return PipelineConfig(
        input_root=Path(args.input) if args.input else None,
        output_root=Path(args.output) if args.output else Path("amc_output"),
    )


def _load_segments(config: PipelineConfig):
    state = _build_state_store(config)
    segs = []
    for row in state.fetch_segments():
        payload = dict(row["payload"])
        payload["status"] = row.get("status", payload.get("status", "preprocessed"))
        segs.append(_segment_from_payload(payload))
    return segs


def _slice_source(source: Path, start_sec: float, end_sec: float, out: Path) -> bool:
    """Write [start_sec, end_sec) of `source` to `out` as wav. soundfile first
    (handles any codec/layout), wave fallback for 16-bit PCM."""
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(source))
        sr = info.samplerate
        start = max(0, int(round(start_sec * sr)))
        stop = min(info.frames, int(round(end_sec * sr)))
        if stop <= start:
            return False
        data, _ = sf.read(str(source), start=start, stop=stop, dtype="int16", always_2d=False)
        sf.write(str(out), data, sr, subtype="PCM_16")
        return True
    except Exception:
        pass
    try:
        import wave

        with wave.open(str(source), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            start = max(0, int(round(start_sec * sr)))
            stop = min(nframes, int(round(end_sec * sr)))
            if stop <= start:
                return False
            wf.setpos(start)
            frames = wf.readframes(stop - start)
        with wave.open(str(out), "wb") as ow:
            ow.setnchannels(nch)
            ow.setsampwidth(sw)
            ow.setframerate(sr)
            ow.writeframes(frames)
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"    (slice failed for {source}: {exc!r})", file=sys.stderr)
        return False


def _to_mp3(wav: Path, ffmpeg: str | None) -> Path | None:
    if not ffmpeg:
        return None
    mp3 = wav.with_suffix(".mp3")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
             "-ac", "1", "-ar", "16000", "-b:a", "32k", str(mp3)],
            check=True,
        )
        return mp3 if mp3.exists() else None
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"    (ffmpeg failed for {wav}: {exc!r})", file=sys.stderr)
        return None


def _emit_b64(path: Path, label: str, max_kb: float) -> None:
    raw = path.read_bytes()
    kb = len(raw) / 1024.0
    if kb > max_kb:
        print(f"    [skip base64 for {label}: {kb:.1f} KB > {max_kb:.0f} KB cap; "
              f"fetch the file directly: {path}]")
        return
    enc = base64.b64encode(raw).decode("ascii")
    print(f"-----BEGIN {label}-----")
    for i in range(0, len(enc), 76):
        print(enc[i:i + 76])
    print(f"-----END {label}-----")


def main() -> int:
    p = argparse.ArgumentParser(description="Export segment audio for listening")
    p.add_argument("--config")
    p.add_argument("--input")
    p.add_argument("--output", required=True)
    p.add_argument("--sid", action="append", default=[], help="segment id (repeatable)")
    p.add_argument("--context-sec", type=float, default=4.0,
                   help="extra seconds of source audio before/after the segment span")
    p.add_argument("--dump-dir", default=None,
                   help="where to write clips (default: <output>/.pii_pipeline/reports/segment_audio)")
    p.add_argument("--emit-b64", action="store_true",
                   help="base64-print the own-clip mp3 (for SSM transfer)")
    p.add_argument("--max-b64-kb", type=float, default=18.0)
    args = p.parse_args()

    if not args.sid:
        print("provide at least one --sid", file=sys.stderr)
        return 2

    config = _load_config(args)
    segs = _load_segments(config)
    if not segs:
        print("no segments found in state; run preprocess first", file=sys.stderr)
        return 2

    by_id = {s.segment_id: s for s in segs}
    # neighbours per call, ordered by start time, for context slicing.
    by_call: dict = {}
    for s in segs:
        key = getattr(s, "call_id", None) or getattr(s, "file_id", None) or s.segment_id
        by_call.setdefault(key, []).append(s)
    for v in by_call.values():
        v.sort(key=lambda s: float(getattr(s, "start_sec", 0.0) or 0.0))

    dump_dir = Path(args.dump_dir) if args.dump_dir else (
        Path(args.output) / ".pii_pipeline" / "reports" / "segment_audio"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("    (ffmpeg not found; exporting wav only -- base64 may exceed the SSM cap)")

    rc = 0
    for sid in args.sid:
        seg = by_id.get(sid)
        if seg is None:
            print(f"\n== {sid} ==\n    NOT FOUND in state", file=sys.stderr)
            rc = 1
            continue
        start = float(getattr(seg, "start_sec", 0.0) or 0.0)
        end = float(getattr(seg, "end_sec", 0.0) or 0.0)
        dur = float(getattr(seg, "duration_sec", max(0.0, end - start)) or 0.0)
        source = Path(getattr(seg, "source_path", ""))
        own = Path(getattr(seg, "segment_audio_path", ""))
        key = getattr(seg, "call_id", None) or getattr(seg, "file_id", None) or sid
        neigh = by_call.get(key, [seg])
        idx = next((i for i, s in enumerate(neigh) if s.segment_id == sid), 0)

        print(f"\n== {sid} ==")
        print(f"    source : {source}")
        print(f"    span   : [{start:.2f}, {end:.2f}] dur={dur:.2f}s")
        print(f"    own clip: {own}")
        lo = neigh[max(0, idx - 2)]
        hi = neigh[min(len(neigh) - 1, idx + 2)]
        ctx_start = max(0.0, float(getattr(lo, "start_sec", start)) - args.context_sec)
        ctx_end = float(getattr(hi, "end_sec", end)) + args.context_sec
        print(f"    neighbours (same call, by time):")
        for j in range(max(0, idx - 2), min(len(neigh), idx + 3)):
            n = neigh[j]
            mark = " <-- target" if n.segment_id == sid else ""
            print(f"        [{float(getattr(n,'start_sec',0)):.2f},{float(getattr(n,'end_sec',0)):.2f}] {n.segment_id}{mark}")

        # own clip
        own_wav = dump_dir / f"{sid}.own.wav"
        if own and own.exists():
            shutil.copyfile(own, own_wav)
        else:
            _slice_source(source, start, end, own_wav)
        own_mp3 = _to_mp3(own_wav, ffmpeg)

        # context slice from the original source
        ctx_wav = dump_dir / f"{sid}.context.wav"
        ok = _slice_source(source, ctx_start, ctx_end, ctx_wav) if source else False
        ctx_mp3 = _to_mp3(ctx_wav, ffmpeg) if ok else None

        print(f"    wrote   : {own_wav}" + (f" , {own_mp3}" if own_mp3 else ""))
        if ok:
            print(f"    context : {ctx_wav}" + (f" , {ctx_mp3}" if ctx_mp3 else "")
                  + f"  [{ctx_start:.2f},{ctx_end:.2f}]")

        if args.emit_b64:
            target = own_mp3 or own_wav
            if target.exists():
                _emit_b64(target, f"{sid} OWN {'MP3' if target.suffix=='.mp3' else 'WAV'}", args.max_b64_kb)

    print(f"\nAll clips under: {dump_dir}")
    if args.emit_b64:
        print("Decode on your Mac, e.g.:")
        print("    pbpaste | sed -n '/-----BEGIN/,/-----END/p' | grep -vE 'BEGIN|END' "
              "| base64 -d > clip.mp3 && afplay clip.mp3")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
