#!/usr/bin/env python3
"""Check a smoke run's outputs against a completed reference run.

Answers the only question that matters before committing a fleet to a year of
audio: did swapping the ASR model change anything OTHER than the ASR model?

Compares the smoke manifest against 2026-full (structure, columns, per-call
artifacts) and, for the calls that were replayed from 2022-full, against that
run's own rows (segmentation and language labels for the same audio).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Column families that are EXPECTED to differ: the run under test votes with
# parakeet where the reference run voted with whisper.
SWAPPED = ("whisper", "parakeet")

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def read_manifest(path: Path, columns: list[str] | None = None) -> dict:
    return pq.read_table(path, columns=columns).to_pydict()


def audio_duration_sec(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=Path, required=True, help="smoke shard output dir")
    ap.add_argument("--reference", type=Path, required=True, help="2026-full shard output dir")
    ap.add_argument("--replay-reference", type=Path, default=None,
                    help="2022-full shard output dir holding the replayed calls")
    ap.add_argument("--input-root", type=Path, required=True, help="subset root fed to the run")
    args = ap.parse_args()

    smoke_parq = args.smoke / "manifests" / "all_segments.parquet"
    ref_parq = args.reference / "manifests" / "all_segments.parquet"
    if not check(smoke_parq.exists(), "smoke manifest exists", str(smoke_parq)):
        return 1

    # --- 1. Column parity, allowing only the whisper -> parakeet swap ---------
    smoke_cols = set(pq.ParquetFile(smoke_parq).schema_arrow.names)
    ref_cols = set(pq.ParquetFile(ref_parq).schema_arrow.names)

    def unswap(cols: set[str]) -> set[str]:
        return {c.replace(f"{SWAPPED[1]}_", f"{SWAPPED[0]}_") for c in cols}

    missing = ref_cols - unswap(smoke_cols)
    extra = unswap(smoke_cols) - ref_cols
    check(not missing and not extra, "manifest columns match 2026-full (modulo whisper->parakeet)",
          f"missing={sorted(missing)} extra={sorted(extra)}")
    check(any(c.startswith("parakeet_") for c in smoke_cols),
          "parakeet transcript columns present")

    # --- 2. Every input call produced rows and a redacted file ----------------
    inputs = {p.name: p.parent.name for p in args.input_root.glob("*/*") if p.is_dir()}
    man = read_manifest(smoke_parq)
    rows_by_call: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(man["call_id"]):
        rows_by_call[cid].append(i)

    check(len(inputs) == len(rows_by_call),
          "every input call has manifest rows",
          f"inputs={len(inputs)} calls_in_manifest={len(rows_by_call)}")
    missing_calls = sorted(set(inputs) - set(rows_by_call))
    if missing_calls:
        notes.append(f"calls with no rows: {missing_calls[:5]}")

    # Redacted audio: one file per call, same layout as the reference run
    # (<shard>/<year>/<call_id>/audio.opus), and the same duration as the source.
    dur_bad, redacted_missing = [], []
    for cid, year in inputs.items():
        out_audio = args.smoke / year / cid / "audio.opus"
        if not out_audio.exists():
            redacted_missing.append(cid)
            continue
        src = args.input_root / year / cid / "audio.opus"
        d_out, d_src = audio_duration_sec(out_audio), audio_duration_sec(src)
        if d_out is None or d_src is None or abs(d_out - d_src) > 0.25:
            dur_bad.append((cid, d_src, d_out))
    check(not redacted_missing, "redacted audio.opus exists for every call",
          f"missing={redacted_missing[:5]}")
    check(not dur_bad, "redacted audio duration matches source (<=0.25s)",
          f"bad={dur_bad[:3]}")

    # Segment WAVs must still be on disk -- downstream training reads these.
    seg_paths = [p for p in man.get("segment_audio_path_abs", []) if p]
    present = sum(1 for p in seg_paths[:400] if Path(p).exists())
    check(seg_paths and present == min(len(seg_paths), 400),
          "segment wavs exist on disk",
          f"checked={min(len(seg_paths), 400)} present={present} total_rows={len(seg_paths)}")

    # --- 3. The four-model panel actually voted ------------------------------
    panel = Counter()
    for present_models in man.get("models_present", []):
        for m in str(present_models or "").replace(";", ",").split(","):
            m = m.strip()
            if m:
                panel[m] += 1
    check("parakeet" in panel and "whisper" not in panel,
          "panel is parakeet-based with no whisper", f"models_present={dict(panel)}")
    agreement = Counter(str(v) for v in man.get("model_agreement", []))
    notes.append(f"model_agreement: {dict(agreement)}")

    # --- 4. Redaction / alignment health ------------------------------------
    align = Counter(str(v) for v in man.get("alignment_status", []))
    check(align.get("failed", 0) == 0, "no failed alignments", f"alignment_status={dict(align)}")
    red = Counter(str(v) for v in man.get("redacted_status", []))
    check(all(k in {"ok", "redacted", "clean", "no_pii", "skipped"} for k in red),
          "redacted_status values healthy", f"redacted_status={dict(red)}")

    pii_rows = [i for i, n in enumerate(man.get("pii_count", [])) if (n or 0) > 0]
    check(bool(pii_rows), "pii detected in the sample", f"segments_with_pii={len(pii_rows)}")
    no_plan = [i for i in pii_rows if not (man["mask_intervals_json"][i] or "").strip("[] \n")]
    check(not no_plan, "every pii segment has mask intervals",
          f"pii_without_intervals={len(no_plan)}")

    # --- 4b. Mask intervals must sit inside their segment and stay targeted ---
    # A mask that covers the whole segment means alignment silently fell back to
    # "beep everything": no PII leaks, but the audio is useless for training.
    out_of_bounds, whole_segment, empty_iv = 0, 0, 0
    for i in pii_rows:
        try:
            intervals = json.loads(man["mask_intervals_json"][i] or "[]")
        except json.JSONDecodeError:
            empty_iv += 1
            continue
        seg_len = float(man["end_sec"][i]) - float(man["start_sec"][i])
        covered = 0.0
        for iv in intervals:
            start, end = (iv[0], iv[1]) if isinstance(iv, (list, tuple)) else (iv["start"], iv["end"])
            start, end = float(start), float(end)
            if end <= start or start < -0.05 or end > seg_len + 0.05:
                out_of_bounds += 1
            covered += max(0.0, end - start)
        if seg_len > 0 and covered >= seg_len * 0.98:
            whole_segment += 1
    check(out_of_bounds == 0, "mask intervals lie inside their segment",
          f"out_of_bounds={out_of_bounds}")
    check(whole_segment <= len(pii_rows) * 0.05,
          "masks are targeted, not whole-segment beeps",
          f"whole_segment={whole_segment}/{len(pii_rows)}")

    # --- 5. Language labels ---------------------------------------------------
    langs = Counter(str(v or "none") for v in man.get("language", []))
    check(langs.get("none", 0) + langs.get("None", 0) == 0,
          "every segment carries a language label", f"languages={dict(langs)}")

    # --- 6. Replayed 2022 calls: same segmentation, same language ------------
    if args.replay_reference:
        ref22 = args.replay_reference / "manifests" / "all_segments.parquet"
        replay_ids = {cid for cid, year in inputs.items() if year == "2022"}
        if ref22.exists() and replay_ids:
            old = read_manifest(ref22, ["call_id", "segment_id", "start_sample",
                                        "end_sample", "language"])
            old_by_call: dict[str, dict[str, tuple]] = defaultdict(dict)
            for cid, sid, ss, es, lang in zip(old["call_id"], old["segment_id"],
                                              old["start_sample"], old["end_sample"],
                                              old["language"]):
                if cid in replay_ids:
                    old_by_call[cid][str(sid)] = (ss, es, lang)

            new_by_call: dict[str, dict[str, tuple]] = defaultdict(dict)
            for i, cid in enumerate(man["call_id"]):
                if cid in replay_ids:
                    new_by_call[cid][str(man["segment_id"][i])] = (
                        man["start_sample"][i], man["end_sample"][i], man["language"][i])

            seg_count_bad, span_bad, lang_agree, lang_total = [], 0, 0, 0
            for cid in sorted(replay_ids):
                o, n = old_by_call.get(cid, {}), new_by_call.get(cid, {})
                if len(o) != len(n):
                    seg_count_bad.append((cid, len(o), len(n)))
                for sid, (ss, es, lang) in o.items():
                    if sid not in n:
                        continue
                    if (n[sid][0], n[sid][1]) != (ss, es):
                        span_bad += 1
                    lang_total += 1
                    lang_agree += int(str(n[sid][2] or "")[:2] == str(lang or "")[:2])
            check(not seg_count_bad, "replayed 2022 calls segment identically",
                  f"count_mismatches={seg_count_bad}")
            check(span_bad == 0, "replayed segment spans identical", f"mismatched_spans={span_bad}")
            rate = (lang_agree / lang_total * 100) if lang_total else 0.0
            check(lang_total > 0 and rate >= 95.0,
                  "parakeet language labels agree with whisper on replayed calls",
                  f"{lang_agree}/{lang_total} = {rate:.2f}%")
            es_new = sum(1 for cid in replay_ids
                         for v in (t[2] for t in new_by_call.get(cid, {}).values())
                         if str(v or "").startswith("es"))
            notes.append(f"replayed spanish segments labelled es by parakeet: {es_new}")

    print()
    for n in notes:
        print(f"note: {n}")
    print()
    if failures:
        print(f"VERDICT: FAIL ({len(failures)} checks) -> {failures}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
