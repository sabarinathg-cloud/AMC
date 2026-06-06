#!/usr/bin/env python3
"""Inspect a single call end-to-end: per-model transcripts, consensus, detected PII, the
mask plan, and (optionally) proof that the redacted audio is actually altered exactly at the
masked intervals and untouched everywhere else.

All the textual data already lives in the merged manifest (one row per segment), produced by
the manifest stage. The audio proof loads the source + redacted files and compares them.

Usage (on any host that can see the shared mount):
    python ops/inspect_call.py --manifest /mnt/amc-data/amc-redacted-final/smoke-20/manifests \
        [--call-id <id>] [--segments 8] [--no-audio]

--manifest accepts: an all_segments.jsonl file, a directory containing all_segments.jsonl,
a parquet directory (manifests/parquet), or a single .parquet file.
If --call-id is omitted, the call with the most detected PII spans is chosen (most useful demo).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MODELS = ["whisper", "qwen", "cohere", "granite"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(manifest: Path) -> list[dict[str, Any]]:
    if manifest.is_dir():
        jsonl = manifest / "all_segments.jsonl"
        if jsonl.exists():
            return _read_jsonl(jsonl)
        # else assume a parquet dataset directory
        import pandas as pd  # noqa: WPS433

        return pd.read_parquet(manifest).to_dict("records")
    if manifest.suffix == ".jsonl":
        return _read_jsonl(manifest)
    if manifest.suffix == ".parquet":
        import pandas as pd  # noqa: WPS433

        return pd.read_parquet(manifest).to_dict("records")
    return _read_jsonl(manifest)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_field(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def pick_call(rows: list[dict[str, Any]], call_id: str | None) -> str:
    if call_id:
        return call_id
    pii_by_call: dict[str, int] = defaultdict(int)
    seg_by_call: dict[str, int] = defaultdict(int)
    for r in rows:
        cid = str(r.get("call_id", ""))
        pii_by_call[cid] += _as_int(r.get("pii_count"))
        seg_by_call[cid] += 1
    if not pii_by_call:
        raise SystemExit("no rows found in manifest")
    # Prefer the call with the most PII (best demo); tie-break on segment count.
    return max(pii_by_call, key=lambda c: (pii_by_call[c], seg_by_call[c]))


def _fmt_spans(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "(none)"
    parts = []
    for s in spans:
        et = s.get("entity_type", "?")
        txt = s.get("text", "")
        a, b = s.get("start", "?"), s.get("end", "?")
        conf = s.get("confidence", "")
        src = s.get("source", "")
        cstr = f" conf={float(conf):.2f}" if conf not in ("", None) else ""
        parts.append(f'[{et} "{txt}" chars {a}-{b}{cstr} via {src}]')
    return " ".join(parts)


def _fmt_intervals(intervals: list[dict[str, Any]], seg_start: float) -> str:
    if not intervals:
        return "(none)"
    parts = []
    for i in intervals:
        i0 = _as_float(i.get("start_sec"))
        i1 = _as_float(i.get("end_sec"))
        et = i.get("entity_type") or i.get("reason") or "mask"
        # global time on the full call recording
        parts.append(f"[{seg_start + i0:.2f}-{seg_start + i1:.2f}s {et}]")
    return " ".join(parts)


def print_report(rows: list[dict[str, Any]], call_id: str, max_segments: int) -> list[dict[str, Any]]:
    segs = [r for r in rows if str(r.get("call_id", "")) == call_id]
    segs.sort(key=lambda r: _as_float(r.get("start_sec")))
    total_pii = sum(_as_int(r.get("pii_count")) for r in segs)
    src = next((r.get("source_path_abs") for r in segs if r.get("source_path_abs")), "?")
    red = next((r.get("redacted_audio_path_abs") for r in segs if r.get("redacted_audio_path_abs")), "?")

    print("=" * 100)
    print(f"CALL {call_id}")
    print(f"  segments={len(segs)}  total_PII_spans={total_pii}")
    print(f"  source  : {src}")
    print(f"  redacted: {red}")
    print("=" * 100)

    shown = segs[:max_segments] if max_segments > 0 else segs
    for r in shown:
        t0 = _as_float(r.get("start_sec"))
        t1 = _as_float(r.get("end_sec"))
        lang = r.get("language", "")
        method = r.get("consensus_method", "")
        score = r.get("consensus_score", "")
        selected = r.get("selected_model", "")
        print(f"\n── segment {r.get('segment_id')}  [{t0:.2f}-{t1:.2f}s]  lang={lang}  "
              f"consensus={method} score={score} selected={selected}")
        for m in MODELS:
            txt = r.get(f"{m}_transcript", "")
            err = r.get(f"{m}_error", "")
            if err:
                print(f"    {m:<8}: <ERROR: {err}>")
            else:
                print(f"    {m:<8}: {txt!r}")
        print(f"    {'final':<8}: {r.get('final_transcript', '')!r}")
        print(f"    PII     : {_fmt_spans(_parse_json_field(r.get('pii_spans_json')))}")
        print(f"    MASK    : {_fmt_intervals(_parse_json_field(r.get('mask_intervals_json')), t0)}")
    if max_segments > 0 and len(segs) > max_segments:
        print(f"\n  ... {len(segs) - max_segments} more segment(s) not shown (use --segments 0 for all)")
    return segs


def _load_mono(path: str):
    import numpy as np

    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1), sr
    except Exception:
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(y, dtype="float32"), sr


def audio_proof(segs: list[dict[str, Any]]) -> None:
    import numpy as np

    src = next((r.get("source_path_abs") for r in segs if r.get("source_path_abs")), None)
    red = next((r.get("redacted_audio_path_abs") for r in segs if r.get("redacted_audio_path_abs")), None)
    if not src or not red or not Path(src).exists() or not Path(red).exists():
        print(f"\n[audio] source/redacted not both present (src={src}, red={red}); skipping audio proof")
        return

    s_audio, s_sr = _load_mono(src)
    r_audio, r_sr = _load_mono(red)
    if s_sr != r_sr:
        import librosa

        r_audio = librosa.resample(r_audio, orig_sr=r_sr, target_sr=s_sr)
        r_sr = s_sr
    n = min(len(s_audio), len(r_audio))
    s_audio, r_audio = s_audio[:n], r_audio[:n]
    sr = s_sr

    masked = np.zeros(n, dtype=bool)
    n_intervals = 0
    for r in segs:
        seg_start = _as_float(r.get("start_sec"))
        for i in _parse_json_field(r.get("mask_intervals_json")):
            g0 = seg_start + _as_float(i.get("start_sec"))
            g1 = seg_start + _as_float(i.get("end_sec"))
            a, b = int(g0 * sr), int(g1 * sr)
            a, b = max(0, a), min(n, b)
            if b > a:
                masked[a:b] = True
                n_intervals += 1

    masked_sec = masked.sum() / sr
    diff = np.abs(r_audio - s_audio)
    masked_diff = float(diff[masked].mean()) if masked.any() else 0.0
    clean_diff = float(diff[~masked].mean()) if (~masked).any() else 0.0
    src_rms_masked = float(np.sqrt((s_audio[masked] ** 2).mean())) if masked.any() else 0.0
    red_rms_masked = float(np.sqrt((r_audio[masked] ** 2).mean())) if masked.any() else 0.0

    print("\n" + "-" * 100)
    print("AUDIO MASKING PROOF")
    print(f"  duration                : {n / sr:.2f}s @ {sr}Hz")
    print(f"  mask intervals applied  : {n_intervals}  ({masked_sec:.2f}s masked)")
    print(f"  |redacted-source| INSIDE masked windows : {masked_diff:.5f}   (should be LARGE)")
    print(f"  |redacted-source| OUTSIDE masked windows: {clean_diff:.6f}   (should be ~0)")
    print(f"  source RMS inside masked windows        : {src_rms_masked:.5f}  (speech energy)")
    print(f"  redacted RMS inside masked windows      : {red_rms_masked:.5f}  (beep/silence energy)")
    if not masked.any():
        print("  VERDICT: no PII intervals for this call -> nothing to mask (expected if call has no PII)")
    elif masked_diff > 1e-4 and clean_diff < masked_diff / 10.0:
        print("  VERDICT: ✅ masking is TARGETED and APPLIED — audio changed only within PII windows")
    else:
        print("  VERDICT: ⚠️  unexpected — review (masked change too small or non-masked regions altered)")
    print("-" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="all_segments.jsonl, a manifests dir, or a parquet dir/file")
    ap.add_argument("--call-id", default=None, help="specific call id (default: the call with the most PII)")
    ap.add_argument("--segments", type=int, default=8, help="max segments to print (0 = all)")
    ap.add_argument("--no-audio", action="store_true", help="skip the source-vs-redacted audio proof")
    args = ap.parse_args()

    rows = load_rows(Path(args.manifest))
    if not rows:
        raise SystemExit(f"no manifest rows under {args.manifest}")
    call_id = pick_call(rows, args.call_id)
    segs = print_report(rows, call_id, args.segments)
    if not args.no_audio:
        audio_proof(segs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
