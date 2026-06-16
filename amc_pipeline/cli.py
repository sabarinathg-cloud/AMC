from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amc-pipeline")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["dry-run", "run", "resume", "retry-failed", "validate", "audit", "clean-cache"]:
        p = sub.add_parser(name)
        _add_common_args(p)
    stage = sub.add_parser("run-stage")
    _add_common_args(stage)
    stage.add_argument("stage")
    pause = sub.add_parser("pause")
    pause.add_argument("--output", type=Path, required=True)
    pause.add_argument("--worker-id", default=None)
    pause.add_argument("--run-id", default="default")
    pause.add_argument("--global", dest="global_pause", action="store_true")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default=None, help="Comma-separated ASR models, for example: whisper,qwen,cohere,granite")
    parser.add_argument("--asr-batch-sizes", default=None, help="Comma-separated ASR batch count caps, for example: qwen=8,cohere=4,granite=4 (dynamic duration budgets still apply)")
    parser.add_argument("--detectors", default=None, help="Comma-separated PII detectors")
    parser.add_argument("--vad-backend", choices=["silero", "energy"], default=None)
    parser.add_argument("--mask-strategy", choices=["beep", "silence", "noise"], default=None)
    parser.add_argument("--allow-fallback-format", choices=["wav"], default=None)
    parser.add_argument("--discovery-hash-mode", choices=["content", "path", "fast", "path_size_mtime", "metadata"], default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--manifest-no-csv", action="store_true", help="Manifest stage: skip the single all_segments.csv (keep JSONL + Parquet). Use at scale.")
    parser.add_argument("--manifest-no-per-year", action="store_true", help="Manifest stage: skip the per-year segments.{jsonl,csv} duplicates. Use at scale.")
    parser.add_argument("--manifest-parquet-batch-size", type=int, default=None, help="Manifest stage: rows per Parquet row group (memory/throughput knob)")
    parser.add_argument("--skip-stage-manifest", action="store_true", help="Skip the intermediate manifest write at the end of a stage (preprocess). Downstream reads the DB and the final manifest stage regenerates it; use at scale to avoid a redundant multi-GB Lustre write.")


def load_cli_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        cfg = PipelineConfig.from_file(args.config, input_root=getattr(args, "input", None), output_root=args.output)
    else:
        cfg = PipelineConfig(input_root=getattr(args, "input", None), output_root=args.output)
    if getattr(args, "run_id", None):
        cfg.run_id = args.run_id
    if getattr(args, "models", None):
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        for name, model_cfg in cfg.asr_models.items():
            model_cfg.enabled = name in wanted
    if getattr(args, "asr_batch_sizes", None):
        _apply_asr_batch_sizes(cfg, args.asr_batch_sizes)
    if getattr(args, "detectors", None):
        wanted = {x.strip() for x in args.detectors.split(",") if x.strip()}
        for name, detector_cfg in cfg.pii_models.items():
            detector_cfg.enabled = name in wanted
    if getattr(args, "vad_backend", None):
        cfg.audio.vad_backend = args.vad_backend
    if getattr(args, "mask_strategy", None):
        cfg.masking.strategy = args.mask_strategy
    if getattr(args, "allow_fallback_format", None) == "wav":
        cfg.masking.allow_wav_fallback = True
    if getattr(args, "discovery_hash_mode", None):
        cfg.discovery.hash_mode = args.discovery_hash_mode
    if getattr(args, "num_shards", None) is not None:
        cfg.discovery.num_shards = args.num_shards
    if getattr(args, "shard_index", None) is not None:
        cfg.discovery.shard_index = args.shard_index
    if getattr(args, "no_progress", False):
        cfg.progress_enabled = False
    if getattr(args, "manifest_no_csv", False):
        cfg.manifest_write_csv = False
    if getattr(args, "manifest_no_per_year", False):
        cfg.manifest_write_per_year = False
    if getattr(args, "manifest_parquet_batch_size", None) is not None:
        cfg.manifest_parquet_batch_size = args.manifest_parquet_batch_size
    if getattr(args, "skip_stage_manifest", False):
        cfg.skip_stage_manifest = True
    return cfg


def _apply_asr_batch_sizes(cfg: PipelineConfig, raw: str) -> None:
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        name, sep, value = item.partition("=")
        if not sep:
            batch_size = int(name)
            for model_cfg in cfg.asr_models.values():
                model_cfg.batch_size = batch_size
            continue
        model_name = name.strip()
        if model_name not in cfg.asr_models:
            raise ValueError(f"Unknown ASR model in --asr-batch-sizes: {model_name}")
        cfg.asr_models[model_name].batch_size = int(value.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_cli_config(args)
        pipeline = Pipeline(cfg)
        if args.command == "dry-run":
            result = pipeline.dry_run()
        elif args.command in {"run", "resume"}:
            result = pipeline.run_all()
        elif args.command == "run-stage":
            result = pipeline.run_stage(args.stage.split(":", 1)[0])
        elif args.command == "pause":
            result = pipeline.pause(worker_id=args.worker_id, global_pause=args.global_pause)
        elif args.command == "clean-cache":
            result = pipeline.clean_cache()
        elif args.command == "validate":
            result = pipeline.validate()
        elif args.command == "audit":
            result = pipeline.validate()
        elif args.command == "retry-failed":
            result = {"command": args.command, "status": "available", "message": "Retry uses the same stage commands; failed scopes remain recorded in state"}
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(f"amc-pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
