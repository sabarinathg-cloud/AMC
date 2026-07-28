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


def num(value) -> float:
    """Manifest numerics are written as strings; treat unparseable as zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def seg_key(segment_id: str) -> str:
    """Channel + index tail of a segment id, e.g. `ch0_00003`.

    The leading part of the id is derived from the input layout, so the same audio
    replayed from a subset directory gets a different prefix. The tail is what
    identifies the segment within a call.
    """
    parts = str(segment_id or "").split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else str(segment_id)


def mask_spans(row_json: str) -> list[tuple[float, float]]:
    try:
        intervals = json.loads(row_json or "[]")
    except json.JSONDecodeError:
        return []
    spans = []
    for iv in intervals:
        if isinstance(iv, (list, tuple)):
            spans.append((num(iv[0]), num(iv[1])))
        else:
            spans.append((num(iv.get("start_sec", iv.get("start"))),
                          num(iv.get("end_sec", iv.get("end")))))
    return spans


def mask_stats(rows: dict, models: tuple[str, ...]) -> dict:
    """Per-run rates that only mean something next to another run's rates.

    How often a mask covers its whole segment, how far a mask runs past the segment
    end (the mask pad does this by design), and how much of the corpus each model
    transcribed are all properties of the corpus and the pipeline, not of the ASR
    swap -- so they are compared against the reference run rather than a guess.
    """
    total = whole = 0
    overshoot = 0.0
    for i, count in enumerate(rows.get("pii_count", [])):
        if num(count) <= 0:
            continue
        seg_len = num(rows["end_sec"][i]) - num(rows["start_sec"][i])
        spans = mask_spans(rows["mask_intervals_json"][i])
        total += 1
        covered = sum(max(0.0, e - s) for s, e in spans)
        whole += int(seg_len > 0 and covered >= seg_len * 0.98)
        for start, end in spans:
            if end > start:
                overshoot = max(overshoot, end - seg_len, -start)
    n = len(rows.get("pii_count", []))
    return {
        "pii_segments": total,
        "whole_segment_rate": whole / max(total, 1),
        "max_overshoot_sec": overshoot,
        "coverage": {
            m: sum(1 for t in rows.get(f"{m}_transcript", []) if (t or "").strip()) / max(n, 1)
            for m in models
        },
    }


def reference_stats(ref_parq: Path, limit: int = 60000) -> dict:
    columns = ["pii_count", "mask_intervals_json", "start_sec", "end_sec"] + [
        f"{m}_transcript" for m in ("whisper", "qwen", "cohere", "granite")
    ]
    rows: dict[str, list] = {c: [] for c in columns}
    seen = 0
    for batch in pq.ParquetFile(ref_parq).iter_batches(batch_size=20000, columns=columns):
        chunk = batch.to_pydict()
        for c in columns:
            rows[c].extend(chunk[c])
        seen += len(chunk["pii_count"])
        if seen >= limit:
            break
    stats = mask_stats(rows, ("whisper", "qwen", "cohere", "granite"))
    # The reference voted with whisper where this run votes with parakeet.
    stats["coverage"]["parakeet"] = stats["coverage"].pop("whisper")
    stats["segments_sampled"] = seen
    return stats


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
    # Only losing a column is a regression. The reference parquet was written by an
    # older build, so columns added since (the agreement/confidence set) show up as
    # extras and are reported rather than failed.
    check(not missing, "no manifest column from 2026-full is missing",
          f"missing={sorted(missing)}")
    if extra:
        notes.append(f"columns added since the reference run: {sorted(extra)}")
    check(any(c.startswith("parakeet_") for c in smoke_cols),
          "parakeet transcript columns present")
    check(not any(c.startswith("whisper_") for c in smoke_cols),
          "no whisper columns remain",
          f"whisper_cols={sorted(c for c in smoke_cols if c.startswith('whisper_'))}")

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
    # `models_present` is a count, so the panel is checked by asking how much of the
    # corpus each model transcribed, against the same figure for the reference run:
    # granite legitimately returns nothing on ~10% of segments in both.
    models = ("parakeet", "qwen", "cohere", "granite")
    smoke_stats = mask_stats(man, models)
    ref_stats = reference_stats(ref_parq)
    notes.append(f"reference sampled {ref_stats['segments_sampled']} segments")
    thin = {
        m: (round(smoke_stats["coverage"][m], 3), round(ref_stats["coverage"][m], 3))
        for m in models
        if smoke_stats["coverage"][m] < ref_stats["coverage"][m] - 0.05
    }
    check(not thin, "each model transcribed as much as it did in the reference run",
          f"smoke={ {m: round(v, 3) for m, v in smoke_stats['coverage'].items()} } "
          f"reference={ {m: round(v, 3) for m, v in ref_stats['coverage'].items()} } "
          f"below_baseline={thin}")
    present_hist = Counter(str(v) for v in man.get("models_present", []))
    notes.append(f"models_present histogram: {dict(present_hist)}")
    notes.append(f"model_agreement: {dict(Counter(str(v) for v in man.get('model_agreement', [])))}")
    notes.append(f"consensus_method: {dict(Counter(str(v) for v in man.get('consensus_method', [])))}")

    # --- 4. Redaction / alignment health ------------------------------------
    align = Counter(str(v) for v in man.get("alignment_status", []))
    check(align.get("failed", 0) == 0, "no failed alignments", f"alignment_status={dict(align)}")
    red = Counter(str(v) for v in man.get("redacted_status", []))
    check(all(k in {"completed", "ok", "redacted", "clean", "no_pii", "skipped"} for k in red),
          "redacted_status values healthy", f"redacted_status={dict(red)}")

    pii_rows = [i for i, n in enumerate(man.get("pii_count", [])) if num(n) > 0]
    check(bool(pii_rows), "pii detected in the sample", f"segments_with_pii={len(pii_rows)}")
    no_plan = [i for i in pii_rows if not (man["mask_intervals_json"][i] or "").strip("[] \n")]
    check(not no_plan, "every pii segment has mask intervals",
          f"pii_without_intervals={len(no_plan)}")

    # --- 4b. Mask intervals must stay in the segment's frame and stay targeted -
    # Intervals are segment-relative and run past the segment edge by the mask pad,
    # which redact clips against the file; the reference run's overshoot is the
    # yardstick. A mask covering the whole segment leaks nothing but is useless for
    # training, so its rate must not exceed the reference run's either.
    inverted = sum(1 for i in pii_rows for s, e in mask_spans(man["mask_intervals_json"][i]) if e <= s)
    check(inverted == 0, "no inverted or empty mask intervals", f"inverted={inverted}")
    check(smoke_stats["max_overshoot_sec"] <= max(ref_stats["max_overshoot_sec"], 0.05) + 0.01,
          "masks overrun the segment edge no more than in the reference run",
          f"smoke={smoke_stats['max_overshoot_sec']:.3f}s "
          f"reference={ref_stats['max_overshoot_sec']:.3f}s (mask pad)")
    check(smoke_stats["whole_segment_rate"] <= ref_stats["whole_segment_rate"] + 0.05,
          "whole-segment masks no more common than in the reference run",
          f"smoke={smoke_stats['whole_segment_rate']*100:.1f}% "
          f"reference={ref_stats['whole_segment_rate']*100:.1f}%")

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
                    # Keyed on the channel+index tail: the id prefix follows the input
                    # layout, which differs for a call replayed out of a subset dir.
                    old_by_call[cid][seg_key(sid)] = (num(ss), num(es), lang)

            new_by_call: dict[str, dict[str, tuple]] = defaultdict(dict)
            for i, cid in enumerate(man["call_id"]):
                if cid in replay_ids:
                    new_by_call[cid][seg_key(man["segment_id"][i])] = (
                        num(man["start_sample"][i]), num(man["end_sample"][i]), man["language"][i])

            seg_count_bad, span_bad = [], 0
            confusion: Counter = Counter()
            # Spanish share per call, old vs new. The label picks the alignment model,
            # so what has to hold is that a Spanish call still comes out Spanish --
            # NOT that the labels match whisper's, which flap per segment (in this very
            # sample whisper called plainly Spanish audio Greek, Finnish and Italian,
            # and rendered "Si, perdon" as "Deep in it").
            spanish_loss = []
            for cid in sorted(replay_ids):
                o, n = old_by_call.get(cid, {}), new_by_call.get(cid, {})
                if len(o) != len(n):
                    seg_count_bad.append((cid, len(o), len(n)))
                matched = [(v, n[sid]) for sid, v in o.items() if sid in n]
                for (ss, es, old_lang), (nss, nes, new_lang) in matched:
                    if (nss, nes) != (ss, es):
                        span_bad += 1
                    confusion[(str(old_lang or "")[:2], str(new_lang or "")[:2])] += 1
                if not matched:
                    spanish_loss.append((cid, "no segments matched"))
                    continue
                old_es = sum(1 for (_, _, l), _ in matched if str(l or "").startswith("es"))
                new_es = sum(1 for _, (_, _, l) in matched if str(l or "").startswith("es"))
                if new_es < old_es - max(1, 0.02 * len(matched)):
                    spanish_loss.append((cid, f"{old_es}->{new_es} of {len(matched)}"))
            check(not seg_count_bad, "replayed 2022 calls segment identically",
                  f"count_mismatches={seg_count_bad}")
            check(span_bad == 0, "replayed segment spans identical", f"mismatched_spans={span_bad}")
            check(not spanish_loss, "no replayed spanish call loses spanish coverage",
                  f"regressions={spanish_loss}")
            agree = sum(n for (a, b), n in confusion.items() if a == b)
            total = sum(confusion.values())
            notes.append(f"replay language agreement with whisper: {agree}/{total}"
                         f" ({agree / max(total, 1) * 100:.1f}%)")
            notes.append("replay language changes: " + ", ".join(
                f"{a}->{b}:{n}" for (a, b), n in confusion.most_common() if a != b))

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
