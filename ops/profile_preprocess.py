#!/usr/bin/env python3
"""Profile the preprocess stage per-file, broken down by sub-step.

Times decode -> read -> VAD (per channel) -> chunk -> segment-write on a sample
of real input files, using the SAME functions the pipeline uses, so we can see
exactly where preprocess wall-time goes and A/B the VAD device.

It writes decoded/segment WAVs to a local scratch dir and never touches the run
DB, so it is safe to run on a live fleet instance.

Usage:
  AMC_VAD_DEVICE=cuda python3 ops/profile_preprocess.py --input <dir> --limit 8
  AMC_VAD_DEVICE=cpu  python3 ops/profile_preprocess.py --input <dir> --limit 8
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from amc_pipeline.audio import decode_to_wav, extract_channel, read_wav, write_mono_segment
from amc_pipeline.config import AudioConfig
from amc_pipeline.segmentation import split_chunks_on_pauses, vad_intervals


AUDIO_EXTS = {".opus", ".ogg", ".mp3", ".m4a", ".wav", ".flac", ".webm"}
# Don't profile our own checkout, venvs, caches, or prior run outputs (segment WAVs
# under amc-runs are not representative real calls).
PRUNE_DIRS = {".git", "venvs", "amc-runs", "amc-cache", ".cache", ".pii_pipeline",
              "huggingface", "torch", "AMC", "node_modules"}


def find_inputs(root: Path, limit: int) -> list[Path]:
    # Early-stopping walk: AMC_IN can be a multi-TB tree, so never materialize/sort it.
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE_DIRS)
        for name in sorted(filenames):
            if Path(name).suffix.lower() in AUDIO_EXTS:
                out.append(Path(dirpath) / name)
                if len(out) >= limit:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/amc-profile"))
    args = ap.parse_args()

    ac = AudioConfig()
    device = os.environ.get("AMC_VAD_DEVICE", "(auto)")
    args.scratch.mkdir(parents=True, exist_ok=True)

    files = find_inputs(args.input, args.limit)
    if not files:
        print(f"no input files under {args.input}")
        return 1
    print(f"VAD device={device}  files={len(files)}  sr={ac.target_sample_rate}")
    print(f"{'file':<28} {'dur_s':>7} {'decode':>7} {'read':>7} {'vad':>8} {'chunk':>7} {'write':>7} {'segs':>5} {'total':>7}")

    tot = {"decode": 0.0, "read": 0.0, "vad": 0.0, "chunk": 0.0, "write": 0.0, "total": 0.0, "dur": 0.0, "segs": 0}
    for i, src in enumerate(files):
        t_all = time.perf_counter()
        dec = args.scratch / f"dec_{i}.wav"
        t = time.perf_counter()
        wp = decode_to_wav(src, dec, ac.target_sample_rate) if src.suffix.lower() != ".wav" else src
        t_decode = time.perf_counter() - t
        t = time.perf_counter()
        buf = read_wav(wp)
        t_read = time.perf_counter() - t
        dur = len(buf.frames) / buf.sample_rate if buf.sample_rate else 0.0
        t_vad = t_chunk = t_write = 0.0
        segs = 0
        for ch in range(buf.channels):
            samples = extract_channel(buf, ch)
            t = time.perf_counter()
            intervals = vad_intervals(
                samples, buf.sample_rate, backend=ac.vad_backend,
                threshold_ratio=ac.vad_threshold_ratio, window_ms=ac.vad_window_ms,
                merge_gap_sec=ac.merge_gap_sec, silero_repo_or_dir=ac.silero_repo_or_dir, merge=False,
            )
            t_vad += time.perf_counter() - t
            if not intervals and len(samples):
                intervals = [(0.0, len(samples) / buf.sample_rate)]
            t = time.perf_counter()
            chunks = split_chunks_on_pauses(
                intervals, len(samples) / buf.sample_rate, target_sec=ac.target_segment_sec,
                max_sec=ac.max_segment_sec, min_sec=ac.min_segment_sec,
                min_split_gap_sec=ac.min_split_gap_sec, merge_gap_sec=ac.merge_gap_sec,
                max_recursion=ac.max_split_recursion,
            )
            t_chunk += time.perf_counter() - t
            t = time.perf_counter()
            for idx, (s, e) in enumerate(chunks):
                ss = max(0, int(round(s * buf.sample_rate)))
                es = min(len(samples), int(round(e * buf.sample_rate)))
                if es <= ss:
                    continue
                write_mono_segment(args.scratch / f"seg_{i}_{ch}_{idx}.wav", samples[ss:es], buf.sample_rate)
                segs += 1
            t_write += time.perf_counter() - t
        total = time.perf_counter() - t_all
        for k, v in (("decode", t_decode), ("read", t_read), ("vad", t_vad), ("chunk", t_chunk), ("write", t_write), ("total", total), ("dur", dur)):
            tot[k] += v
        tot["segs"] += segs
        print(f"{src.name[:28]:<28} {dur:>7.1f} {t_decode:>7.2f} {t_read:>7.2f} {t_vad:>8.2f} {t_chunk:>7.2f} {t_write:>7.2f} {segs:>5} {total:>7.2f}")
        for f in args.scratch.glob("*.wav"):
            try:
                f.unlink()
            except Exception:
                pass

    n = len(files)
    print("-" * 96)
    print(f"{'MEAN/file':<28} {tot['dur']/n:>7.1f} {tot['decode']/n:>7.2f} {tot['read']/n:>7.2f} {tot['vad']/n:>8.2f} {tot['chunk']/n:>7.2f} {tot['write']/n:>7.2f} {tot['segs']//n:>5} {tot['total']/n:>7.2f}")
    rtfx = tot["dur"] / tot["total"] if tot["total"] else 0.0
    print(f"audio={tot['dur']:.0f}s  wall={tot['total']:.1f}s  RTFx={rtfx:.2f}  (VAD share={100*tot['vad']/tot['total']:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
