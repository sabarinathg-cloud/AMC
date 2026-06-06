#!/usr/bin/env python3
"""ASR batching/dtype parity harness (run on the GPU host with model weights).

Purpose
-------
Prove the throughput optimizations do not change transcripts before they are
trusted in production. For a chosen model and a sample of already-preprocessed
segments, it compares:

  1. batch=1 (cap=1, prefetch off)  vs  dynamic batching (duration budget)
  2. for Cohere only: float32  vs  bfloat16

and reports exact-match rate and word error rate (WER) per comparison. Cohere's
bf16 default should only be enabled (set ``asr_models.cohere.dtype: bfloat16``)
once this harness shows negligible change against float32.

This script CANNOT run on a machine without CUDA and the local model weights.
It reads segments from the pipeline state DB produced by the ``preprocess`` stage.

Examples
--------
    python3 ops/asr_parity_check.py \
        --input /mnt/amc-data/input --output /mnt/amc-data/output \
        --model cohere --limit 50

    python3 ops/asr_parity_check.py --config config.yaml --model granite --limit 30
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import ASRModelConfig, PipelineConfig
from amc_pipeline.normalization import normalize_transcript
from amc_pipeline.pipeline import _segment_from_payload
from amc_pipeline.state import PostgresStateStore, SQLiteStateStore
from amc_pipeline.transcription import CohereAdapter, GraniteAdapter, QwenAdapter, WhisperAdapter


ADAPTERS = {"qwen": QwenAdapter, "cohere": CohereAdapter, "granite": GraniteAdapter}


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


def _load_segments(config: PipelineConfig, limit: int):
    state = _build_state_store(config)
    segments = []
    for row in state.fetch_segments():
        payload = dict(row["payload"])
        payload["status"] = row.get("status", payload.get("status", "preprocessed"))
        segments.append(_segment_from_payload(payload))
    segments.sort(key=lambda s: (s.duration_sec, s.segment_id))
    if limit and limit > 0:
        segments = segments[:limit]
    return segments


def _run(adapter, segments) -> dict[str, str]:
    adapter.progress_enabled = True
    t0 = time.time()
    try:
        adapter.preflight()
        results = adapter.transcribe_batch(segments)
    finally:
        try:
            adapter.close()
        except Exception:
            pass
    elapsed = time.time() - t0
    audio = sum(max(0.0, float(getattr(s, "duration_sec", 0.0) or 0.0)) for s in segments)
    rtfx = (audio / elapsed) if elapsed > 0 else 0.0
    print(f"    elapsed={elapsed:.1f}s  audio={audio:.1f}s  RTFx={rtfx:.1f}x")
    return {r.segment_id: (r.transcript or "") for r in results}


def _wer(reference: str, hypothesis: str) -> float:
    ref = normalize_transcript(reference).split()
    hyp = normalize_transcript(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein distance over word tokens.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(hyp)] / len(ref)


def _compare(label: str, baseline: dict[str, str], candidate: dict[str, str]) -> None:
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        print(f"[{label}] no overlapping segments to compare")
        return
    exact = sum(1 for sid in ids if baseline[sid] == candidate[sid])
    norm_exact = sum(1 for sid in ids if normalize_transcript(baseline[sid]) == normalize_transcript(candidate[sid]))
    wers = [_wer(baseline[sid], candidate[sid]) for sid in ids]
    mean_wer = sum(wers) / len(wers)
    max_wer = max(wers)
    print(f"[{label}] segments={len(ids)}")
    print(f"    exact-match (raw):        {exact}/{len(ids)} = {exact / len(ids):.4f}")
    print(f"    exact-match (normalized): {norm_exact}/{len(ids)} = {norm_exact / len(ids):.4f}")
    print(f"    mean WER: {mean_wer:.5f}   max WER: {max_wer:.5f}")
    worst = sorted(zip(ids, wers), key=lambda kv: kv[1], reverse=True)[:3]
    for sid, w in worst:
        if w > 0:
            print(f"    worst sid={sid} wer={w:.4f}")
            print(f"        baseline : {baseline[sid]!r}")
            print(f"        candidate: {candidate[sid]!r}")


def _cfg_for(model: str, base: ASRModelConfig, **overrides) -> ASRModelConfig:
    data = dataclasses.asdict(base)
    data.update(overrides)
    return ASRModelConfig(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR batching/dtype parity check")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model", required=True, choices=sorted({*ADAPTERS, "whisper"}))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--budget", type=float, default=None, help="audio-seconds budget for the dynamic run")
    parser.add_argument("--max-batch", type=int, default=None, help="count cap for the dynamic run")
    parser.add_argument("--check-bf16", action="store_true", help="Cohere only: also compare float32 vs bfloat16")
    parser.add_argument(
        "--compare-path",
        help="Whisper only: a second model dir to A/B against the configured one, "
        "both on the WER-safe per-segment path (e.g. large-v3 vs large-v3-turbo).",
    )
    args = parser.parse_args()

    config = _load_config(args)
    base = config.asr_models.get(args.model)
    if base is None or not base.path:
        print(f"model '{args.model}' is not configured with a path", file=sys.stderr)
        return 2
    segments = _load_segments(config, args.limit)
    if not segments:
        print("no segments found in state; run the preprocess stage first", file=sys.stderr)
        return 2
    print(f"model={args.model} segments={len(segments)} (path={base.path})")

    # Whisper has no dynamic-batching axis. Two A/B modes:
    #   default            : per-segment (WER-safe) vs packed (fast, opt-in)
    #   --compare-path PATH : per-segment large-v3 vs per-segment PATH (e.g. turbo)
    if args.model == "whisper":
        def _mk(batched: bool, path: str | None = None) -> WhisperAdapter:
            adapter = WhisperAdapter(path or base.path or "", base.batch_size or 0, base.device)
            adapter.batched = batched
            return adapter

        if args.compare_path:
            print(f"\n== model A: per-segment (path={base.path}) ==")
            baseline = _run(_mk(False), segments)
            print(f"== model B: per-segment (path={args.compare_path}) ==")
            candidate = _run(_mk(False, args.compare_path), segments)
            _compare("whisper_modelA_vs_modelB_persegment", baseline, candidate)
            print(
                "\nNote: this compares two MODELS on the same WER-safe per-segment path.\n"
                "Adopt model B (e.g. large-v3-turbo) if RTFx is clearly higher AND drift\n"
                "is benign (spacing/casing/punctuation), with NO dropped or invented words.\n"
                "True WER vs ground truth needs labeled references, not this drift check."
            )
            return 0

        print("\n== running whisper per-segment (AMC_WHISPER_BATCHED=0 baseline) ==")
        baseline = _run(_mk(False), segments)
        print("== running whisper packed (AMC_WHISPER_BATCHED=1, opt-in) ==")
        packed = _run(_mk(True), segments)
        _compare("whisper_persegment_vs_packed", baseline, packed)
        print(
            "\nNote: packing adds cross-segment context/silence, so drift is EXPECTED.\n"
            "Per-segment is the default because packing can hallucinate filler on very\n"
            "short clips. Validate worst-drift examples are real re-wordings, not\n"
            "empty/hallucinated text. True WER needs labeled references, not this check."
        )
        return 0

    adapter_cls = ADAPTERS[args.model]

    # 1) batch=1 baseline (cap=1, prefetch off) vs dynamic batching.
    batch1_cfg = _cfg_for(args.model, base, batch_size=1, max_batch_size=1, batch_audio_sec_budget=None, prefetch=False)
    dynamic_cfg = _cfg_for(args.model, base, max_batch_size=args.max_batch, batch_audio_sec_budget=args.budget)

    print("\n== running batch=1 baseline ==")
    baseline = _run(adapter_cls(batch1_cfg), segments)
    print("== running dynamic batching ==")
    dynamic = _run(adapter_cls(dynamic_cfg), segments)
    _compare("batch1_vs_dynamic", baseline, dynamic)

    # 2) Cohere dtype parity (float32 baseline vs bfloat16).
    if args.model == "cohere" and args.check_bf16:
        print("\n== running cohere float32 ==")
        fp32 = _run(adapter_cls(_cfg_for(args.model, base, dtype="float32", max_batch_size=args.max_batch, batch_audio_sec_budget=args.budget)), segments)
        print("== running cohere bfloat16 ==")
        bf16 = _run(adapter_cls(_cfg_for(args.model, base, dtype="bfloat16", max_batch_size=args.max_batch, batch_audio_sec_budget=args.budget)), segments)
        _compare("cohere_fp32_vs_bf16", fp32, bf16)

    print("\nDone. Keep bf16 / larger batches only if exact-match is ~1.0 and WER is negligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
