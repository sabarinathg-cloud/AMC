#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import PipelineConfig
from amc_pipeline.discovery import (
    _belongs_to_shard,
    _iter_audio_paths,
    infer_call_id,
    normalized_hash_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the sharded AMC input file count, signature, and total file count.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-mode", default="path")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def compute_signature(input_root: Path, output_root: Path, hash_mode: str, num_shards: int, shard_index: int) -> tuple[int, str, int]:
    # Single filesystem walk that yields BOTH the total discovered count (across all shards)
    # and this shard's slice. The signature is computed from the shard's (relative_path, size)
    # pairs -- identical to the prior implementation -- so existing stage markers stay valid.
    # We intentionally avoid building full AudioFileRecord objects / content fingerprints here:
    # this runs at the start of every stage and only needs counts + a stable signature.
    config = PipelineConfig(input_root=input_root, output_root=output_root)
    normalized_hash_mode(hash_mode)  # validates the mode string early
    root = config.input_root.resolve()
    exts = {e.lower() for e in config.supported_extensions}

    total = 0
    shard_entries: list[tuple[str, int]] = []
    for path in _iter_audio_paths(root, exts):
        total += 1
        rel = path.relative_to(root)
        call_id = infer_call_id(path, rel)
        if not _belongs_to_shard(call_id, num_shards, shard_index):
            continue
        shard_entries.append((rel.as_posix(), path.stat().st_size))

    shard_entries.sort()
    payload = "\n".join(f"{rel}:{size}" for rel, size in shard_entries)
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return len(shard_entries), signature, total


def main() -> int:
    args = parse_args()
    count, signature, total = compute_signature(args.input, args.output, args.hash_mode, args.num_shards, args.shard_index)
    # Output contract: "<shard_file_count> <signature> <total_file_count>".
    # run_shard_no_docker.sh reads all three; older callers reading two fields still work.
    print(count, signature, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
