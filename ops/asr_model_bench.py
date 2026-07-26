#!/usr/bin/env python3
"""Benchmark an ASR model/config against the rest of the consensus panel.

WHY THIS EXISTS
  We want to replace Whisper with something faster, under a hard "accuracy must
  not drop" constraint. Published WER (Open ASR Leaderboard) is measured on
  audiobooks, podcasts and meetings -- not 8kHz-origin call-centre telephony cut
  into ~4s VAD clips. So the decision has to be made on OUR audio.

  We have no human ground truth, but we do have four independent transcripts per
  segment. That is enough: for any model X, the other panel members form an
  independent reference. If the candidate lands CLOSER to the rest of the panel
  than Whisper does, on the same segments, that is evidence its accuracy is at
  least as good.

  The reference deliberately EXCLUDES both Whisper and the candidate, so the
  comparison is not biased toward the incumbent. (Scoring against
  `normalized_final_transcript` would be biased -- Whisper voted in it.)

METRICS
  panel_wer      Corpus WER of the model against each reference model, pooled.
                 The headline number: lower = more consistent with the panel.
  agreed_wer     Corpus WER on the high-confidence subset where >=2 reference
                 models produced the SAME normalized transcript. Closest
                 available proxy for true WER.
  exact_match    Fraction of the high-confidence subset matched exactly.
  rtfx           Audio-seconds transcribed per wall-second (higher = faster).
  seg_s          Segments per wall-second.

USAGE
  # Score a candidate (runs the model; needs its venv + a GPU)
  /mnt/amc-data/venvs/parakeet/bin/python ops/asr_model_bench.py \
      --run-root /mnt/amc-data/amc-runs/2022-full --model parakeet --limit 2000

  # Whisper config variants (Track A): same harness, main venv
  AMC_WHISPER_BEAM_SIZE=1 /mnt/amc-data/venvs/main/bin/python \
      ops/asr_model_bench.py --run-root ... --model whisper --limit 2000

  # No GPU / no model: just score what is already in the manifests
  python3 ops/asr_model_bench.py --run-root ... --baseline-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.config import PipelineConfig  # noqa: E402
from amc_pipeline.consensus import DEFAULT_AGREEMENT_MODELS  # noqa: E402
from amc_pipeline.models import SegmentRecord  # noqa: E402
from amc_pipeline.normalization import normalize_transcript  # noqa: E402


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over word lists."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


class Corpus:
    """Accumulates corpus-level WER (total edits / total reference words).

    Corpus WER, not the mean of per-utterance WERs -- the latter over-weights
    one-word backchannel clips, which are ~a third of this dataset.
    """

    def __init__(self) -> None:
        self.errors = 0
        self.words = 0
        self.n = 0
        self.exact = 0

    def add(self, ref: str, hyp: str) -> None:
        r, h = ref.split(), hyp.split()
        self.errors += _edits(r, h)
        self.words += len(r)
        self.n += 1
        self.exact += int(ref == hyp)

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else float("nan")

    @property
    def exact_rate(self) -> float:
        return self.exact / self.n if self.n else float("nan")


def _majority(values: list[str]) -> str | None:
    """The transcript >=2 reference models agree on exactly, else None."""
    for v in values:
        if v and values.count(v) >= 2:
            return v
    return None


def score_model(rows: list[dict], norms: dict[str, str], reference_models: list[str]) -> dict[str, Any]:
    """Score one model's normalized transcripts against the reference models.

    `norms` maps segment_id -> normalized transcript for the model under test.
    """
    panel = Corpus()
    agreed = Corpus()
    for row in rows:
        hyp = norms.get(row["segment_id"])
        if hyp is None:
            continue
        refs = [row["norm"][m] for m in reference_models if row["norm"].get(m)]
        if not refs:
            continue
        for ref in refs:
            panel.add(ref, hyp)
        consensus = _majority(refs) if len(refs) >= 2 else None
        if consensus is not None:
            agreed.add(consensus, hyp)
    return {
        "panel_wer": panel.wer,
        "panel_pairs": panel.n,
        "agreed_wer": agreed.wer,
        "agreed_exact_match": agreed.exact_rate,
        "agreed_segments": agreed.n,
        "empty_transcripts": sum(1 for r in rows if not (norms.get(r["segment_id"]) or "").strip()),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_rows(manifests: list[Path], models: list[str], limit: int, seed: int) -> list[dict]:
    """Randomly sample segments across shards, reading only whole row groups.

    Reading 21M rows x ~15 text columns to draw a few thousand samples would cost
    far more than the benchmark itself, so we pull a random row group per shard
    and sample within it.
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    columns = [
        "segment_id", "call_id", "file_id", "year", "channel", "language",
        "duration_sec", "start_sec", "end_sec", "start_sample", "end_sample",
        "duration_samples", "sample_rate", "segment_audio_path_abs", "source_path_abs",
        "model_agreement",
    ]
    for m in models:
        columns += [f"{m}_transcript", f"{m}_normalized_transcript", f"{m}_error"]

    per_file = max(1, limit // max(1, len(manifests)))
    out: list[dict] = []
    for path in manifests:
        pf = pq.ParquetFile(path)
        available = [c for c in columns if c in pf.schema_arrow.names]
        groups = list(range(pf.num_row_groups))
        if not groups:
            continue
        rng.shuffle(groups)
        picked: list[dict] = []
        for g in groups:
            table = pf.read_row_group(g, columns=available)
            rows = table.to_pylist()
            rng.shuffle(rows)
            picked.extend(rows)
            if len(picked) >= per_file:
                break
        for row in picked[:per_file]:
            row["norm"] = {
                m: str(row.get(f"{m}_normalized_transcript") or "").strip()
                for m in models
                if not str(row.get(f"{m}_error") or "").strip()
                and str(row.get(f"{m}_transcript") or "").strip()
            }
            out.append(row)
    rng.shuffle(out)
    return out[:limit]


def to_segment(row: dict) -> SegmentRecord:
    return SegmentRecord(
        segment_id=row["segment_id"],
        file_id=str(row.get("file_id") or ""),
        call_id=str(row.get("call_id") or ""),
        year=str(row.get("year") or ""),
        channel=int(row.get("channel") or 0),
        source_path=Path(str(row.get("source_path_abs") or "")),
        segment_audio_path=Path(str(row.get("segment_audio_path_abs") or "")),
        start_sec=float(row.get("start_sec") or 0.0),
        end_sec=float(row.get("end_sec") or 0.0),
        start_sample=int(row.get("start_sample") or 0),
        end_sample=int(row.get("end_sample") or 0),
        duration_sec=float(row.get("duration_sec") or 0.0),
        duration_samples=int(row.get("duration_samples") or 0),
        sample_rate=int(row.get("sample_rate") or 16000),
        language=row.get("language") or "en",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: float, pct: bool = True) -> str:
    if value != value:  # NaN
        return "     n/a"
    return f"{value * 100:7.2f}%" if pct else f"{value:8.2f}"


def print_report(report: dict[str, Any]) -> None:
    ref = report["reference_models"]
    print()
    print("=" * 78)
    print(f"ASR PANEL BENCHMARK  --  {report['segments']:,} segments, "
          f"{report['audio_sec'] / 3600:.2f} h audio")
    print(f"reference panel: {', '.join(ref)}   (candidate and whisper excluded)")
    print("=" * 78)
    header = f"{'model':<22}{'panel_wer':>11}{'agreed_wer':>12}{'exact':>10}{'n_agreed':>10}"
    print(header)
    print("-" * 78)
    for name, s in report["scores"].items():
        print(f"{name:<22}{_fmt(s['panel_wer']):>11}{_fmt(s['agreed_wer']):>12}"
              f"{_fmt(s['agreed_exact_match']):>10}{s['agreed_segments']:>10,}")
    context = report.get("panel_context")
    if context:
        print()
        print("panel context (each judged by the other reference models -- smaller")
        print("reference set, so for scale only, not comparable to the rows above):")
        for name, s in context.items():
            print(f"  {name:<20}{_fmt(s['panel_wer']):>11}{_fmt(s['agreed_wer']):>12}"
                  f"{_fmt(s['agreed_exact_match']):>10}{s['agreed_segments']:>10,}")
    timing = report.get("timing")
    if timing:
        print("-" * 78)
        print(f"speed: {timing['wall_sec']:.1f}s wall  |  {timing['seg_s']:.1f} seg/s  |  "
              f"RTFx {timing['rtfx']:.1f}")
        base = report.get("whisper_rtfx")
        if base:
            print(f"       {timing['rtfx'] / base:.1f}x Whisper's measured RTFx ({base:.1f})")
    verdict = report.get("verdict")
    if verdict:
        print("-" * 78)
        print(verdict)
    print("=" * 78)
    print()


def build_verdict(candidate: str, scores: dict[str, dict]) -> str:
    cand, whis = scores.get(candidate), scores.get("whisper")
    if not cand or not whis or cand["panel_wer"] != cand["panel_wer"]:
        return ""
    delta = cand["panel_wer"] - whis["panel_wer"]
    rel = (delta / whis["panel_wer"] * 100) if whis["panel_wer"] else 0.0
    if delta < 0:
        return (f"PASS: {candidate} is {abs(rel):.1f}% closer to the panel than whisper "
                f"({cand['panel_wer'] * 100:.2f}% vs {whis['panel_wer'] * 100:.2f}% panel WER). "
                "Accuracy does not drop.")
    return (f"REVIEW: {candidate} is {rel:.1f}% further from the panel than whisper "
            f"({cand['panel_wer'] * 100:.2f}% vs {whis['panel_wer'] * 100:.2f}% panel WER). "
            "Do not swap on this evidence.")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="Existing run dir, e.g. /mnt/amc-data/amc-runs/2022-full")
    ap.add_argument("--model", default=None, help="Candidate ASR model to run (parakeet, whisper, ...)")
    ap.add_argument("--model-path", default=None, help="Override the candidate's model path")
    ap.add_argument("--batch-size", type=int, default=None, help="Override the candidate's batch cap")
    ap.add_argument("--limit", type=int, default=2000, help="Segments to sample (default 2000)")
    ap.add_argument("--seed", type=int, default=1234, help="Sampling seed (keep fixed across variants)")
    ap.add_argument("--shards", default=None, help="Comma-separated shard indices to sample from (default: all)")
    ap.add_argument("--baseline-only", action="store_true", help="Score existing manifest columns; run nothing")
    ap.add_argument("--label", default=None, help="Name for the candidate row (default: the model name)")
    ap.add_argument("--json-out", default=None, help="Write the full report as JSON here")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    pattern = str(run_root / "outputs" / "shard-*" / "manifests" / "all_segments.parquet")
    manifests = sorted(Path(p) for p in glob.glob(pattern))
    if args.shards:
        keep = {s.strip() for s in args.shards.split(",") if s.strip()}
        manifests = [p for p in manifests if p.parts[-3].removeprefix("shard-") in keep]
    if not manifests:
        print(f"no manifests matched {pattern}", file=sys.stderr)
        return 2

    panel = list(DEFAULT_AGREEMENT_MODELS)
    print(f"sampling {args.limit:,} segments from {len(manifests)} manifest(s)...", flush=True)
    rows = sample_rows(manifests, panel, args.limit, args.seed)
    if not rows:
        print("sample came back empty", file=sys.stderr)
        return 2
    audio_sec = sum(float(r.get("duration_sec") or 0.0) for r in rows)

    candidate = args.model if not args.baseline_only else None
    label = args.label or candidate
    # Reference excludes the incumbent AND the candidate so neither is scored
    # against a panel it helped define.
    reference_models = [m for m in panel if m not in {"whisper", candidate}]

    def norms_from_manifest(model: str) -> dict[str, str]:
        return {r["segment_id"]: r["norm"][model] for r in rows if model in r["norm"]}

    # The incumbent is scored against exactly the reference set the candidate will
    # be scored against, which is what makes the two rows directly comparable.
    scores: dict[str, dict] = {"whisper": score_model(rows, norms_from_manifest("whisper"), reference_models)}

    # Context only: each reference model judged by the OTHER reference models
    # (never itself, which would score a free 0). Its reference set is smaller
    # than the candidate's, so these are a scale reference, not a ranking.
    context = {
        m: score_model(rows, norms_from_manifest(m), [x for x in reference_models if x != m])
        for m in reference_models
    }

    report: dict[str, Any] = {
        "run_root": str(run_root),
        "segments": len(rows),
        "audio_sec": audio_sec,
        "reference_models": reference_models,
        "seed": args.seed,
    }

    if candidate:
        cfg = PipelineConfig()
        model_cfg = cfg.asr_models.get(candidate)
        if model_cfg is None:
            print(f"unknown model '{candidate}'; known: {sorted(cfg.asr_models)}", file=sys.stderr)
            return 2
        if args.model_path:
            model_cfg.path = args.model_path
        if args.batch_size:
            model_cfg.batch_size = args.batch_size
        model_cfg.enabled = True

        from amc_pipeline.transcription import _build_single_adapter

        adapter = _build_single_adapter(candidate, model_cfg)
        adapter.progress_enabled = True
        adapter.preflight()
        segments = [to_segment(r) for r in rows]

        # Warm the weights and any lazy kernels before timing.
        print(f"warming up {candidate}...", flush=True)
        adapter.transcribe_batch(segments[: min(16, len(segments))])

        print(f"transcribing {len(segments):,} segments with {candidate}...", flush=True)
        start = time.perf_counter()
        results = adapter.transcribe_batch(segments)
        wall = time.perf_counter() - start
        adapter.close()

        errors = [r for r in results if r.error]
        norms = {
            r.segment_id: normalize_transcript(r.transcript, remove_fillers=True)
            for r in results
            if not r.error
        }
        scores[label] = score_model(rows, norms, reference_models)
        report["timing"] = {
            "wall_sec": wall,
            "seg_s": len(segments) / wall if wall else 0.0,
            "rtfx": audio_sec / wall if wall else 0.0,
        }
        report["candidate_errors"] = len(errors)
        if errors:
            report["candidate_error_sample"] = [e.error for e in errors[:5]]
        report["verdict"] = build_verdict(label, scores)
        report["env"] = {
            k: v for k, v in os.environ.items() if k.startswith("AMC_WHISPER") or k.startswith("AMC_ASR")
        }

    report["scores"] = scores
    report["panel_context"] = context
    print_report(report)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
