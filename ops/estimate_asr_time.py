#!/usr/bin/env python3
"""Project ASR wall-clock for a large run from a completed sample run.

It reads the per-segment audio durations from a finished run's state DBs to learn the
real segments/file and audio/file distribution, then extrapolates to --target-files and
divides total audio by the measured per-model RTFx (audio-seconds per wall-second) and the
GPU count. ASR runs the four models sequentially per shard, so the combined throughput is
the harmonic combination of the per-model RTFx. Stdlib only.

Defaults for RTFx come from ops/asr_batch_sweep.py at cap=128 on an A10G. Override any of
them if your sweep numbers differ.

Example
-------
    python3 ops/estimate_asr_time.py --run-root /mnt/amc-data/amc-runs/2026-smoke-cohere \
        --target-files 500000 --gpus 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys


# Per-model RTFx (audio-seconds processed per wall-second) at cap=128, A10G sweep.
DEFAULT_RTFX = {"whisper": 1.21, "qwen": 7.65, "granite": 8.43, "cohere": 11.76}


def _candidate_dbs(args) -> list[str]:
    if args.db:
        return [args.db]
    root = args.run_root or "/mnt/amc-data/amc-runs"
    suffix = os.path.join("outputs", "shard-*", ".pii_pipeline", "state", "pipeline.sqlite3")
    dbs, seen = [], set()
    for pat in (os.path.join(root, suffix), os.path.join(root, "*", suffix), os.path.join(root, "*", "*", suffix)):
        for db in glob.glob(pat):
            if db not in seen and os.path.exists(db):
                seen.add(db)
                dbs.append(db)
    return sorted(dbs)


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    if not d and not h:
        parts.append(f"{s}s")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Project ASR wall-clock from a sample run")
    ap.add_argument("--db")
    ap.add_argument("--run-root")
    ap.add_argument("--target-files", type=int, default=500000)
    ap.add_argument("--gpus", type=int, default=4, help="number of GPUs (1 per instance on this fleet)")
    for m, r in DEFAULT_RTFX.items():
        ap.add_argument(f"--rtfx-{m}", type=float, default=r, help=f"{m} RTFx (default {r})")
    args = ap.parse_args()

    dbs = _candidate_dbs(args)
    if not dbs:
        print("FATAL: no state DB found. Pass --db or --run-root.", file=sys.stderr)
        return 2

    files: set[str] = set()
    n_segments = 0
    total_audio = 0.0
    for db in dbs:
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except Exception as exc:
            print(f"WARN: cannot open {db}: {exc}", file=sys.stderr)
            continue
        for (pj,) in con.execute("select payload_json from segments"):
            try:
                p = json.loads(pj or "{}")
            except Exception:
                continue
            n_segments += 1
            fid = p.get("file_id")
            if fid:
                files.add(fid)
            total_audio += float(p.get("duration_sec") or 0.0)
        con.close()

    n_files = len(files)
    if n_files == 0 or n_segments == 0:
        print("FATAL: sample run has no segments/files.", file=sys.stderr)
        return 3

    segs_per_file = n_segments / n_files
    audio_per_file = total_audio / n_files
    avg_seg = total_audio / n_segments

    rtfx = {m: getattr(args, f"rtfx_{m}") for m in DEFAULT_RTFX}
    # Sequential models -> wall seconds to push 1 audio-second through all four, on one GPU.
    wall_per_audio_sec = sum(1.0 / r for r in rtfx.values())
    combined_rtfx = 1.0 / wall_per_audio_sec

    target = args.target_files
    proj_audio = audio_per_file * target
    proj_segments = segs_per_file * target

    print("=" * 64)
    print(f"SAMPLE  : {args.run_root or args.db}")
    print(f"  files={n_files}  segments={n_segments}  audio={total_audio/3600:.2f} h")
    print(f"  segments/file={segs_per_file:.1f}  audio/file={audio_per_file:.1f}s  avg_segment={avg_seg:.2f}s")
    print("=" * 64)
    print(f"PROJECTION to {target:,} files:")
    print(f"  segments ~= {proj_segments/1e6:.2f} M")
    print(f"  audio    ~= {proj_audio/3600:.0f} h  ({proj_audio/3600/24:.1f} days of audio)")
    print("-" * 64)
    print("Per-model RTFx (audio-s / wall-s), cap=128:")
    for m, r in rtfx.items():
        gpu_h = proj_audio / r / 3600.0
        print(f"  {m:<8} RTFx={r:>6.2f}   1-GPU={_fmt_hms(gpu_h*3600)}   /{args.gpus}gpu={_fmt_hms(gpu_h*3600/args.gpus)}")
    print("-" * 64)
    one_gpu = proj_audio * wall_per_audio_sec
    fleet = one_gpu / max(1, args.gpus)
    print(f"COMBINED 4-model ASR (sequential): RTFx={combined_rtfx:.2f}")
    print(f"  1 GPU      : {_fmt_hms(one_gpu)}")
    print(f"  {args.gpus} GPUs    : {_fmt_hms(fleet)}")
    print("=" * 64)
    print("Note: sweep benchmarked the LONGEST segments, so real RTFx (full duration mix) is")
    print("usually HIGHER -> these are conservative (upper-bound) wall-clock estimates. ASR only;")
    print("preprocess/PII/align/redact are additional but mostly overlap or are far cheaper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
