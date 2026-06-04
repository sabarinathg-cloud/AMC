#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small AMC call subset with the original year/call/audio layout.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_year = args.source_root / args.year
    output_year = args.output_root / args.year
    if not source_year.exists():
        raise SystemExit(f"source year folder not found: {source_year}")
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if output_year.exists() and args.force:
        shutil.rmtree(output_year)
    output_year.mkdir(parents=True, exist_ok=True)

    # Stream directory entries and stop as soon as we have `--limit` calls.
    # Years can hold ~500K call folders on shared storage; sorting/stat-ing the
    # whole listing first would stall for minutes. os.scandir is lazy and we
    # break early, so this is O(limit), not O(total calls).
    audio_files = []
    with os.scandir(source_year) as entries:
        for entry in entries:
            if len(audio_files) >= args.limit:
                break
            try:
                if not entry.is_dir(follow_symlinks=True):
                    continue
            except OSError:
                continue
            audio = _first_audio(Path(entry.path))
            if audio is None:
                continue
            audio_files.append(audio)
    # Stable ordering of just the small selected set (cheap; keeps manifest deterministic).
    audio_files.sort(key=lambda p: p.parent.name)

    rows = []
    for index, audio in enumerate(audio_files):
        call_id = audio.parent.name
        target_dir = output_year / call_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / audio.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.mode == "copy":
            shutil.copy2(audio, target)
        else:
            target.symlink_to(audio)
        rows.append(
            {
                "index": index,
                "year": args.year,
                "call_id": call_id,
                "source_audio": str(audio),
                "subset_audio": str(target),
                "mode": args.mode,
            }
        )

    manifest_json = args.output_root / "subset_manifest.json"
    manifest_csv = args.output_root / "subset_manifest.csv"
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "year", "call_id", "source_audio", "subset_audio", "mode"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"calls": len(rows), "output_root": str(args.output_root), "manifest_json": str(manifest_json), "manifest_csv": str(manifest_csv)}, indent=2))
    return 0


def _first_audio(call_dir: Path) -> Path | None:
    for name in ["audio.opus", "audio.wav", "audio.mp3", "audio.m4a", "audio.flac", "audio.ogg", "audio.webm"]:
        candidate = call_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(call_dir.glob("audio.*"))
    return matches[0] if matches else None


if __name__ == "__main__":
    raise SystemExit(main())
