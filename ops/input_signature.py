#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import PipelineConfig
from amc_pipeline.discovery import discover_audio_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the sharded AMC input file count and signature.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-mode", default="path")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def compute_signature(input_root: Path, output_root: Path, hash_mode: str, num_shards: int, shard_index: int) -> tuple[int, str]:
    config = PipelineConfig(input_root=input_root, output_root=output_root)
    config.discovery.hash_mode = hash_mode
    config.discovery.num_shards = num_shards
    config.discovery.shard_index = shard_index
    records = sorted(discover_audio_files(config), key=lambda record: record.relative_path.as_posix())
    payload = "\n".join(f"{record.relative_path.as_posix()}:{record.size_bytes}" for record in records)
    return len(records), hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def main() -> int:
    args = parse_args()
    count, signature = compute_signature(args.input, args.output, args.hash_mode, args.num_shards, args.shard_index)
    print(count, signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
