#!/usr/bin/env python3
"""Build a self-contained page for listening to what the masking actually did.

Reviewing whole calls does not scale and does not focus on the risk: what has to
be verified is the few seconds around each mask. So for every mask interval this
cuts the SAME window from the original and from the redacted file and puts the two
players side by side, next to the transcript text that caused the mask.

Calls where a model produced a looping transcript are listed first: those are the
ones where the wrong tie-break would have shipped audio nobody had read, so they
are the segments whose masking most needs a human ear.

    python3 ops/build_review_bundle.py \
      --run /mnt/amc-data/amc-runs/2025-smoke30/outputs/shard-0 \
      --out /mnt/amc-data/amc-runs/2025-smoke30/review

The bundle is a directory of opus clips plus index.html; sync it down and open it.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

COLUMNS = [
    "call_id", "segment_id", "year", "language", "start_sec", "end_sec",
    "final_transcript", "selected_model", "consensus_method", "pii_count",
    "pii_spans_json", "mask_intervals_json", "source_path_abs",
    "redacted_audio_path_abs", "source_duration_sec",
]


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_json(raw, default):
    try:
        out = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError):
        return default
    return out if out else default


def span_text(span: dict, transcript: str) -> tuple[str, str]:
    """(label, quoted text) for a PII span, tolerating either key naming."""
    label = str(span.get("label") or span.get("entity") or span.get("type")
                or span.get("entity_type") or "PII")
    if str(span.get("source") or span.get("detector") or "") == "spoken_number":
        label += " (spoken)"
    text = span.get("text") or span.get("value") or ""
    if not text:
        start = span.get("start_char", span.get("start"))
        end = span.get("end_char", span.get("end"))
        if isinstance(start, int) and isinstance(end, int):
            text = transcript[start:end]
    return label, str(text)


def cut(src: Path, dest: Path, start: float, duration: float) -> bool:
    """Extract `duration` seconds of `src` from `start` as opus."""
    if dest.exists():
        return True
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-accurate_seek",
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
           "-i", str(src), "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return dest.exists() and dest.stat().st_size > 0


def player(rel: str | None) -> str:
    if not rel:
        return "<span class=missing>clip failed</span>"
    return f"<audio controls preload=none src='{html.escape(rel)}'></audio>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="shard output dir")
    ap.add_argument("--out", type=Path, required=True, help="bundle dir to create")
    ap.add_argument("--input-root", type=Path, default=None,
                    help="source audio root, if source_path_abs no longer resolves")
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--max-clips-per-call", type=int, default=6)
    ap.add_argument("--pad", type=float, default=2.5, help="seconds of context each side")
    ap.add_argument("--full-calls", type=int, default=5,
                    help="also copy this many whole redacted calls")
    ap.add_argument("--no-prefer-spoken", dest="prefer_spoken", action="store_false",
                    help="rank by looping models only, ignoring spoken-number masks")
    args = ap.parse_args()

    manifest = args.run / "manifests" / "all_segments.parquet"
    man = pq.read_table(manifest, columns=COLUMNS).to_pydict()

    # Segments where a model looped: reviewed first, since those are the ones the
    # degeneracy guard rescued from shipping unread.
    report = args.run / ".pii_pipeline" / "reports" / "degenerate_asr.json"
    looped = defaultdict(list)
    for row in parse_json(report.read_text() if report.exists() else "", []):
        looped[row["segment_id"]].append(row["model"])

    rows_by_call: dict[str, list[int]] = defaultdict(list)
    for i, call_id in enumerate(man["call_id"]):
        rows_by_call[call_id].append(i)

    def spoken_spans(i: int) -> int:
        """Spans found by the spoken-number detector, which are the newest masks."""
        return str(man["pii_spans_json"][i] or "").count("spoken_number")

    def call_rank(call_id: str) -> tuple:
        rows = rows_by_call[call_id]
        spoken = sum(spoken_spans(i) for i in rows)
        loops = sum(1 for i in rows if man["segment_id"][i] in looped)
        pii = sum(num(man["pii_count"][i]) for i in rows)
        # Numbers read out digit by digit rank first: they were shipping in the clear
        # until the spoken-number detector landed, so they are the masks least proven
        # by ear -- both that the digits are gone and that the speech around them is not.
        return (-spoken, -loops, -pii) if args.prefer_spoken else (-loops, -pii)

    calls = sorted(rows_by_call, key=call_rank)[: args.max_calls]

    clips_dir = args.out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    total_clips = failed_clips = 0

    for call_id in calls:
        rows = sorted(rows_by_call[call_id], key=lambda i: num(man["start_sec"][i]))
        redacted = Path(man["redacted_audio_path_abs"][rows[0]] or "")
        source = Path(man["source_path_abs"][rows[0]] or "")
        if args.input_root and not source.exists():
            source = args.input_root / str(man["year"][rows[0]]) / call_id / source.name
        loops = sorted({m for i in rows for m in looped.get(man["segment_id"][i], [])})

        clip_rows: list[str] = []
        made = 0
        for i in rows:
            if made >= args.max_clips_per_call:
                break
            seg_start = num(man["start_sec"][i])
            transcript = man["final_transcript"][i] or ""
            spans = parse_json(man["pii_spans_json"][i], [])
            intervals = parse_json(man["mask_intervals_json"][i], [])
            if not intervals:
                continue
            labels = [span_text(s, transcript) for s in spans if isinstance(s, dict)]
            for k, iv in enumerate(intervals):
                if made >= args.max_clips_per_call:
                    break
                start = num(iv.get("start_sec", iv.get("start")))
                end = num(iv.get("end_sec", iv.get("end")))
                if end <= start:
                    continue
                # Intervals are segment-relative; clips are cut from the whole call.
                at = seg_start + start
                window = (end - start) + 2 * args.pad
                stem = f"{call_id[:8]}_{man['segment_id'][i].split('_')[-2]}_{k:02d}"
                red_clip = clips_dir / f"{stem}_masked.opus"
                src_clip = clips_dir / f"{stem}_original.opus"
                ok_red = redacted.exists() and cut(redacted, red_clip, at - args.pad, window)
                ok_src = source.exists() and cut(source, src_clip, at - args.pad, window)
                total_clips += 1
                failed_clips += int(not (ok_red and ok_src))
                shown = ", ".join(f"<b>{html.escape(l)}</b>: {html.escape(t)}"
                                  for l, t in labels[:4] if t) or "&mdash;"
                loop_tag = ("<span class=loop>loop rejected: "
                            + ", ".join(looped[man["segment_id"][i]]) + "</span>"
                            if man["segment_id"][i] in looped else "")
                clip_rows.append(
                    "<tr>"
                    f"<td class=t>{at:.1f}s<br><span class=dim>{end - start:.1f}s masked</span></td>"
                    f"<td>{player(f'clips/{src_clip.name}') if ok_src else player(None)}</td>"
                    f"<td>{player(f'clips/{red_clip.name}') if ok_red else player(None)}</td>"
                    f"<td class=tx>{shown}<div class=dim>{html.escape(transcript[:220])}</div>"
                    f"{loop_tag}</td>"
                    "</tr>"
                )
                made += 1

        if not clip_rows:
            continue
        pii_total = int(sum(num(man["pii_count"][i]) for i in rows))
        langs = sorted({str(man["language"][i] or "?") for i in rows})
        cards.append(
            f"<section><h2>{html.escape(call_id)}"
            + (f" <span class=loop>{len(loops)} model loop(s): {', '.join(loops)}</span>" if loops else "")
            + "</h2>"
            f"<div class=dim>{len(rows)} segments &middot; {pii_total} PII spans &middot; "
            f"{num(man['source_duration_sec'][rows[0]]):.0f}s call &middot; lang {'/'.join(langs)} &middot; "
            f"won by {html.escape(str(man['selected_model'][rows[0]] or '?'))}</div>"
            "<table><tr><th>at</th><th>original</th><th>masked</th>"
            "<th>what should be gone</th></tr>"
            + "".join(clip_rows) + "</table></section>"
        )

    # A few whole calls too: clips prove the mask landed, a full call proves the
    # rest of the audio survived it.
    full_dir = args.out / "full_calls"
    full_dir.mkdir(exist_ok=True)
    full_links = []
    for call_id in calls[: args.full_calls]:
        src = Path(man["redacted_audio_path_abs"][rows_by_call[call_id][0]] or "")
        if not src.exists():
            continue
        dest = full_dir / f"{call_id}.opus"
        if not dest.exists():
            shutil.copy2(src, dest)
        full_links.append(
            f"<li><code>{html.escape(call_id)}</code> {player(f'full_calls/{dest.name}')}</li>")

    style = """body{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;max-width:1180px;color:#111}
h1{font-size:20px}h2{font-size:15px;margin:22px 0 2px;font-family:ui-monospace,monospace}
section{border-top:1px solid #e3e3e3;padding-top:10px;margin-top:18px}
table{border-collapse:collapse;width:100%;margin-top:8px}
th{text-align:left;font-size:11px;text-transform:uppercase;color:#666;padding:4px 6px}
td{padding:6px;border-top:1px solid #f0f0f0;vertical-align:top}
td.t{white-space:nowrap;font-variant-numeric:tabular-nums;width:78px}
td.tx{font-size:13px}.dim{color:#777;font-size:12px}
audio{height:32px;width:250px}
.loop{background:#fff3cd;color:#7a5b00;padding:1px 6px;border-radius:3px;font-size:11px}
.missing{color:#b00}ul{padding-left:18px}"""

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Masking review</title>"
        f"<style>{style}</style>"
        f"<h1>Masking review &mdash; {html.escape(args.run.parent.parent.name)}</h1>"
        f"<p class=dim>{len(cards)} calls, {total_clips} masked moments"
        + (f", {failed_clips} clip(s) could not be cut" if failed_clips else "")
        + ". For each mask you hear the same window twice: the original, then what ships. "
        "The PII should be audible on the left and gone on the right. "
        "Calls where a model produced a looping transcript are listed first.</p>"
        + "".join(cards)
        + ("<section><h2>whole redacted calls</h2><ul>" + "".join(full_links) + "</ul></section>"
           if full_links else "")
    )
    print(f"bundle: {args.out}")
    print(f"  calls={len(cards)} clips={total_clips} failed={failed_clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
