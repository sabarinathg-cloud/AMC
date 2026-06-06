#!/usr/bin/env python3
"""ASR batch-size sweep: find the largest batch that is fast, in-memory, and lossless.

Run this on the GPU host (with model weights + the right venv) AFTER the preprocess
stage has populated the state DB with segments. For a chosen model it sweeps a list of
batch-count caps (and optional audio-seconds budgets) and, for each, reports:

  * throughput   -- segments/sec and RTFx (audio-seconds processed per wall-second)
  * peak GPU mem -- max allocated + max reserved (MiB), and headroom vs the device
  * OOM splits   -- how many times the adapter hit CUDA-OOM and auto-split this run
                    (the adapters silently recover by halving, so a too-big batch does
                    NOT crash -- it just stops getting faster and wastes time splitting;
                    num_ooms > 0 is the signal that you have gone past the real ceiling)
  * transcript drift (with --verify) -- exact-match rate + WER vs the smallest cap, so
                    you only adopt a bigger batch if the output is byte-for-byte the same.

It then prints a recommendation: the largest cap with zero OOM splits, comfortable memory
headroom, and identical transcripts.

IMPORTANT venv note: cohere must run in the `cohere` venv, the others in `main`:
    /mnt/amc-data/venvs/main/bin/python   ops/asr_batch_sweep.py --model qwen   ...
    /mnt/amc-data/venvs/cohere/bin/python ops/asr_batch_sweep.py --model cohere ...

Examples
--------
    # Does 64 actually help Qwen on this A10G? Sweep and prove it is lossless.
    python3 ops/asr_batch_sweep.py --input /mnt/amc-data/input --output /mnt/amc-data/output \
        --model qwen --limit 128 --caps 8,16,32,48,64,96 --verify

    # Whisper (faster-whisper internal batch_size) sweep.
    python3 ops/asr_batch_sweep.py --input ... --output ... --model whisper --caps 16,32,48,64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # ops/  -> reuse parity helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # repo root -> amc_pipeline

from asr_parity_check import _cfg_for, _load_config, _load_segments, _wer  # noqa: E402
from amc_pipeline.normalization import normalize_transcript  # noqa: E402
from amc_pipeline.transcription import (  # noqa: E402
    CohereAdapter,
    GraniteAdapter,
    QwenAdapter,
    WhisperAdapter,
)

ADAPTERS = {"qwen": QwenAdapter, "cohere": CohereAdapter, "granite": GraniteAdapter}


def _torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _build_adapter(model: str, base, cap: int, budget: float | None):
    if model == "whisper":
        # faster-whisper takes a single internal batch_size; no audio-seconds budget.
        return WhisperAdapter(base.path or "", cap, base.device)
    cfg = _cfg_for(model, base, batch_size=cap, max_batch_size=cap, batch_audio_sec_budget=budget)
    return ADAPTERS[model](cfg)


def _measure(adapter, segments, torch) -> dict:
    cuda = bool(torch and torch.cuda.is_available())
    ooms_before = 0
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ooms_before = torch.cuda.memory_stats().get("num_ooms", 0)
    adapter.progress_enabled = False
    try:
        adapter.preflight()
        t0 = time.perf_counter()
        results = adapter.transcribe_batch(segments)
        if cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    finally:
        try:
            adapter.close()
        except Exception:
            pass
    out = {
        "wall": wall,
        "peak_alloc_mb": 0.0,
        "peak_resv_mb": 0.0,
        "ooms": 0,
        "transcripts": {r.segment_id: (r.transcript or "") for r in results},
    }
    if cuda:
        out["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        out["peak_resv_mb"] = torch.cuda.max_memory_reserved() / 2**20
        out["ooms"] = torch.cuda.memory_stats().get("num_ooms", 0) - ooms_before
        torch.cuda.empty_cache()
    return out


def _drift(baseline: dict[str, str], candidate: dict[str, str]) -> tuple[float, float]:
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        return (0.0, 1.0)
    norm_exact = sum(1 for sid in ids if normalize_transcript(baseline[sid]) == normalize_transcript(candidate[sid]))
    mean_wer = sum(_wer(baseline[sid], candidate[sid]) for sid in ids) / len(ids)
    return (norm_exact / len(ids), mean_wer)


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR batch-size sweep (throughput + memory + losslessness)")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model", required=True, choices=["whisper", "qwen", "cohere", "granite"])
    parser.add_argument("--limit", type=int, default=128, help="number of (longest-first) segments to benchmark")
    parser.add_argument("--caps", default="8,16,32,48,64", help="comma-separated batch-count caps to sweep")
    parser.add_argument("--budget", type=float, default=None, help="fixed audio-seconds budget for every cap (default: model default)")
    parser.add_argument("--verify", action="store_true", help="compare each cap's transcripts to the smallest cap (losslessness)")
    args = parser.parse_args()

    config = _load_config(args)
    base = config.asr_models.get(args.model)
    if base is None or not base.path:
        print(f"model '{args.model}' is not configured with a path", file=sys.stderr)
        return 2

    # Benchmark the LONGEST segments first: worst case for memory, so a cap that survives
    # here is safe for the whole shard.
    segments = _load_segments(config, args.limit)
    if not segments:
        print("no segments found in state; run the preprocess stage first", file=sys.stderr)
        return 2
    segments.sort(key=lambda s: (s.duration_sec, s.segment_id), reverse=True)
    audio_sec = sum(float(getattr(s, "duration_sec", 0.0) or 0.0) for s in segments)

    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    torch = _torch()
    total_mb = 0.0
    if torch and torch.cuda.is_available():
        total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        print(f"device={torch.cuda.get_device_name(0)} total={total_mb:.0f} MiB")
    print(f"model={args.model} segments={len(segments)} audio={audio_sec:.1f}s path={base.path}")
    print(f"caps={caps} budget={args.budget if args.budget is not None else 'model-default'}\n")

    header = f"{'cap':>5} {'wall_s':>8} {'seg/s':>8} {'RTFx':>8} {'peak_alloc':>11} {'peak_resv':>10} {'OOMsplit':>9}"
    if args.verify:
        header += f" {'exact':>7} {'WER':>7}"
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    baseline_tx: dict[str, str] | None = None
    for cap in caps:
        try:
            res = _measure(_build_adapter(args.model, base, cap, args.budget), segments, torch)
        except Exception as exc:  # a hard failure (not the caught CUDA-OOM path) -> stop climbing
            print(f"{cap:>5}  FAILED: {type(exc).__name__}: {exc}")
            break
        seg_s = len(segments) / res["wall"] if res["wall"] else 0.0
        rtfx = audio_sec / res["wall"] if res["wall"] else 0.0
        line = f"{cap:>5} {res['wall']:>8.2f} {seg_s:>8.2f} {rtfx:>8.2f} {res['peak_alloc_mb']:>10.0f}M {res['peak_resv_mb']:>9.0f}M {res['ooms']:>9d}"
        row = {"cap": cap, "seg_s": seg_s, "rtfx": rtfx, "peak_resv_mb": res["peak_resv_mb"], "ooms": res["ooms"], "exact": 1.0, "wer": 0.0}
        if args.verify:
            if baseline_tx is None:
                baseline_tx = res["transcripts"]
            exact, wer = _drift(baseline_tx, res["transcripts"])
            row["exact"], row["wer"] = exact, wer
            line += f" {exact:>7.4f} {wer:>7.4f}"
        print(line)
        rows.append(row)

    # Recommendation: largest cap that did NOT trip an OOM split, keeps memory headroom, and
    # (if verified) is byte-identical to the smallest cap.
    safe = [
        r for r in rows
        if r["ooms"] == 0
        and (not total_mb or r["peak_resv_mb"] <= 0.92 * total_mb)
        and (not args.verify or (r["exact"] >= 0.999 and r["wer"] <= 1e-4))
    ]
    print()
    if safe:
        best = max(safe, key=lambda r: (round(r["rtfx"], 2), r["cap"]))
        biggest = max(safe, key=lambda r: r["cap"])
        print(f"RECOMMEND cap={best['cap']} (RTFx={best['rtfx']:.2f}, peak_resv={best['peak_resv_mb']:.0f} MiB"
              + (f" / {total_mb:.0f} MiB" if total_mb else "") + ")")
        if biggest["cap"] != best["cap"]:
            print(f"  (largest safe cap is {biggest['cap']}, but {best['cap']} gives the best throughput; "
                  "bigger stopped helping)")
        flag = "--asr-batch-sizes" if args.model != "whisper" else "asr_models.whisper.batch_size"
        print(f"  apply via: {flag} {args.model}={best['cap']}  (whisper: set the config field)")
    else:
        print("No cap was both OOM-free and lossless. Stay at or below the smallest cap, "
              "or lower the audio-seconds budget.")
    print("\nNotes: OOMsplit>0 means the adapter caught CUDA-OOM and auto-split (no crash, but "
          "wasted work) -- treat that cap as over the ceiling. Benchmarked longest segments first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
