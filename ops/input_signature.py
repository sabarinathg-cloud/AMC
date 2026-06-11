#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import PipelineConfig
from amc_pipeline.discovery import discovery_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the sharded AMC input file count, signature, and total file count.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-mode", default="path")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def compute_signature(input_root: Path, output_root: Path, hash_mode: str, num_shards: int, shard_index: int) -> tuple[int, str, int]:
    # Returns (shard_file_count, signature, total_file_count). The signature is the
    # sha256 of this shard's sorted (relative_path, size) pairs -- identical to the
    # prior implementation -- so existing stage markers stay valid. discovery_counts
    # uses the shared discovery cache when available (one tree walk per run instead
    # of one per stage per box) and otherwise falls back to a live walk.
    config = PipelineConfig(input_root=input_root, output_root=output_root)
    config.discovery.hash_mode = hash_mode
    config.discovery.num_shards = num_shards
    config.discovery.shard_index = shard_index
    return discovery_counts(config)


def main() -> int:
    args = parse_args()
    count, signature, total = compute_signature(args.input, args.output, args.hash_mode, args.num_shards, args.shard_index)
    # Output contract: "<shard_file_count> <signature> <total_file_count>".
    # run_shard_no_docker.sh reads all three; older callers reading two fields still work.
    print(count, signature, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
