#!/usr/bin/env python3
"""Inspect ONE segment end-to-end so you can confirm the pipeline actually worked:

  * every ASR model's raw transcript  + the consensus "final" transcript
  * the PII spans that were detected   (entity, text, offsets, which detector)
  * the per-file mask plan intervals   (what audio ranges get bleeped)
  * the redacted output file            (exists? validation ok? duration preserved?)
  * optionally, cut the segment's time window from BOTH the original call and the
    redacted call (--dump-audio) so you can A/B listen and HEAR the masking.

Stdlib only (sqlite3/json/glob/subprocess) so it runs in ANY venv on the box.

Examples
--------
    # Auto-find a shard DB and pick a segment that has PII (so masking is visible):
    python3 ops/inspect_example.py

    # A specific run/shard, dump listenable clips next to the script output:
    python3 ops/inspect_example.py --out /mnt/amc-data/amc-runs/smoke-50/outputs/shard-0 --dump-audio /tmp/amc_listen

    # A specific segment id:
    python3 ops/inspect_example.py --segment-id 2023_000123_ch0_00004
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

SEP = "=" * 78
SUB = "-" * 78


def _connect_ro(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _candidate_dbs(args) -> list[str]:
    if args.db:
        return [args.db]
    if args.out:
        return [str(Path(args.out) / ".pii_pipeline" / "state" / "pipeline.sqlite3")]
    roots = []
    if args.run_root:
        roots.append(args.run_root)
    else:
        runs_root = os.environ.get("AMC_RUNS_ROOT") or "/mnt/amc-data/amc-runs"
        roots.append(runs_root)
    suffix = os.path.join("outputs", "shard-*", ".pii_pipeline", "state", "pipeline.sqlite3")
    dbs: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for pat in (os.path.join(root, "*", suffix), os.path.join(root, suffix), os.path.join(root, "*", "*", suffix)):
            for db in glob.glob(pat):
                if db not in seen and os.path.exists(db):
                    seen.add(db)
                    dbs.append(db)
    # Most segments first -> most likely to have a complete example.
    def _nseg(db: str) -> int:
        try:
            con = _connect_ro(db)
            n = con.execute("select count(*) from segments").fetchone()[0]
            con.close()
            return int(n)
        except Exception:
            return 0
    return sorted(dbs, key=_nseg, reverse=True)


def _artifact(con: sqlite3.Connection, artifact_id: str) -> dict | None:
    row = con.execute("select * from artifacts where artifact_id = ?", (artifact_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["payload"] = json.loads(d.get("payload_json") or "{}")
    return d


def _segment_payload(con: sqlite3.Connection, segment_id: str) -> dict | None:
    row = con.execute("select * from segments where segment_id = ?", (segment_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["payload_json"] or "{}")


def _pick_segment(con: sqlite3.Connection, want_pii: bool) -> str | None:
    if want_pii:
        # First segment whose pii artifact has at least one span.
        for row in con.execute("select artifact_id, payload_json from artifacts where kind='pii'"):
            payload = json.loads(row["payload_json"] or "{}")
            if payload.get("spans"):
                sid = payload.get("segment_id") or row["artifact_id"].split(":", 1)[-1]
                return sid
    # Otherwise any segment that has model results.
    row = con.execute("select segment_id from model_results limit 1").fetchone()
    if row:
        return row["segment_id"]
    row = con.execute("select segment_id from segments limit 1").fetchone()
    return row["segment_id"] if row else None


def _find_db_for_segment(dbs: list[str], segment_id: str) -> str | None:
    for db in dbs:
        try:
            con = _connect_ro(db)
            hit = con.execute("select 1 from segments where segment_id=?", (segment_id,)).fetchone()
            con.close()
            if hit:
                return db
        except Exception:
            continue
    return None


def _fmt_spans(spans: list[dict]) -> str:
    if not spans:
        return "    (none)"
    out = []
    for s in spans:
        out.append(
            "    [{etype}] {text!r}  @char {a}-{b}  conf={c}  via={det}/{src}".format(
                etype=s.get("entity_type"),
                text=s.get("text"),
                a=s.get("start_char"),
                b=s.get("end_char"),
                c=round(float(s.get("confidence", 0) or 0), 3),
                det=s.get("detector_source") or s.get("source"),
                src=s.get("transcript_source", "final"),
            )
        )
    return "\n".join(out)


def _ffmpeg_clip(src: str, start: float, dur: float, dst: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff or not os.path.exists(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.05, dur):.3f}", "-i", src,
           "-ac", "1", "-ar", "16000", "-codec:a", "libmp3lame", "-qscale:a", "5", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return dst.exists()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect one segment: transcripts, PII, masking, redacted audio.")
    ap.add_argument("--db", help="explicit state DB path")
    ap.add_argument("--out", help="shard output dir (<out>/.pii_pipeline/state/pipeline.sqlite3)")
    ap.add_argument("--run-root", help="run root to scan for shard DBs")
    ap.add_argument("--segment-id", help="inspect this exact segment id")
    ap.add_argument("--any", action="store_true", help="don't require PII; just pick any segment")
    ap.add_argument("--dump-audio", metavar="DIR", help="cut original+redacted clips for the segment window into DIR (needs ffmpeg)")
    args = ap.parse_args()

    dbs = _candidate_dbs(args)
    if not dbs:
        print("FATAL: no state DB found. Pass --db or --out or --run-root.", file=sys.stderr)
        return 2

    if args.segment_id:
        db = _find_db_for_segment(dbs, args.segment_id)
        if not db:
            print(f"FATAL: segment {args.segment_id} not found in any candidate DB.", file=sys.stderr)
            return 3
        segment_id = args.segment_id
    else:
        db = dbs[0]
        con = _connect_ro(db)
        segment_id = _pick_segment(con, want_pii=not args.any)
        con.close()
        if not segment_id:
            print(f"FATAL: no segments in {db}", file=sys.stderr)
            return 3

    con = _connect_ro(db)
    seg = _segment_payload(con, segment_id) or {}
    file_id = seg.get("file_id")
    print(SEP)
    print(f"STATE DB : {db}")
    print(f"SEGMENT  : {segment_id}")
    print(f"  call_id={seg.get('call_id')} year={seg.get('year')} channel={seg.get('channel')} file_id={file_id}")
    print(f"  window : {seg.get('start_sec')}s -> {seg.get('end_sec')}s  (dur {seg.get('duration_sec')}s)")
    print(f"  seg wav: {seg.get('segment_audio_path')}")
    print(f"  source : {seg.get('source_path')}")

    # ---- transcripts: every model + consensus final --------------------------------------
    print(SEP)
    print("ASR TRANSCRIPTS (per model)")
    print(SUB)
    rows = con.execute("select model_name, status, payload_json from model_results where segment_id=? order by model_name", (segment_id,)).fetchall()
    if not rows:
        print("  (no model_results)")
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        tx = payload.get("transcript", "")
        conf = payload.get("confidence")
        print(f"  {r['model_name']:<8} [{r['status']}] conf={conf}")
        print(f"      {tx!r}")

    consensus = _artifact(con, f"consensus:{segment_id}")
    print(SUB)
    print("CONSENSUS (final transcript)")
    if consensus:
        cp = consensus["payload"]
        print(f"      {cp.get('final_transcript', '')!r}")
        if cp.get("agreement") is not None:
            print(f"      agreement={cp.get('agreement')} models={cp.get('model_count')}")
    else:
        print("  (no consensus artifact)")

    # ---- PII --------------------------------------------------------------------------------
    pii = _artifact(con, f"pii:{segment_id}")
    print(SEP)
    print("PII SPANS (detected on the final transcript)")
    print(SUB)
    spans = (pii or {}).get("payload", {}).get("spans", []) if pii else []
    print(_fmt_spans(spans))

    # ---- mask plan (per file) + redacted output --------------------------------------------
    print(SEP)
    print("MASK PLAN (audio ranges to bleep, per file)")
    print(SUB)
    mask = _artifact(con, f"mask_plan:{file_id}") if file_id else None
    intervals = (mask or {}).get("payload", {}).get("intervals", []) if mask else []
    seg_intervals = []
    if intervals:
        s0, s1 = float(seg.get("start_sec", 0) or 0), float(seg.get("end_sec", 0) or 0)
        for iv in intervals:
            within = (iv.get("channel") == seg.get("channel")) and not (float(iv.get("end_sec", 0)) <= s0 or float(iv.get("start_sec", 0)) >= s1)
            tag = "  <-- in this segment" if within else ""
            if within:
                seg_intervals.append(iv)
            print(f"    ch{iv.get('channel')}  {float(iv.get('start_sec',0)):.2f}s -> {float(iv.get('end_sec',0)):.2f}s  "
                  f"{iv.get('entity_type')}/{iv.get('reason')} src={iv.get('source')}{tag}")
    else:
        print("    (no mask intervals for this file -> nothing bleeped)")

    redacted = _artifact(con, f"redacted:{file_id}") if file_id else None
    print(SUB)
    print("REDACTED OUTPUT (masked audio)")
    redacted_path = None
    if redacted:
        rp = redacted["payload"]
        redacted_path = rp.get("path")
        exists = bool(redacted_path and os.path.exists(redacted_path))
        val = rp.get("validation", {}) or {}
        print(f"    status   : {redacted.get('status')}")
        print(f"    path     : {redacted_path}   exists={exists}")
        print(f"    validation: ok={val.get('ok')} src_dur={val.get('source_duration')} out_dur={val.get('output_duration')}")
    else:
        print("    (no redacted artifact for this file)")

    con.close()

    # ---- optional: cut listenable clips ----------------------------------------------------
    if args.dump_audio:
        outdir = Path(args.dump_audio)
        s0 = float(seg.get("start_sec", 0) or 0)
        dur = max(0.2, float(seg.get("end_sec", 0) or 0) - s0)
        # pad a little so you hear context around the bleep
        pad = 0.5
        start = max(0.0, s0 - pad)
        wdur = dur + 2 * pad
        print(SEP)
        print(f"AUDIO CLIPS -> {outdir}  (window {start:.2f}s +{wdur:.2f}s)")
        print(SUB)
        made = []
        orig = _ffmpeg_clip(str(seg.get("source_path", "")), start, wdur, outdir / f"{segment_id}__original.mp3")
        if orig:
            made.append(outdir / f"{segment_id}__original.mp3")
        if redacted_path:
            red = _ffmpeg_clip(str(redacted_path), start, wdur, outdir / f"{segment_id}__redacted.mp3")
            if red:
                made.append(outdir / f"{segment_id}__redacted.mp3")
        if not made:
            print("    (ffmpeg missing or source files unavailable on this host)")
        for p in made:
            b64 = base64.b64encode(p.read_bytes()).decode()
            print(f"    {p}  ({p.stat().st_size} bytes)")
            print(f"    BASE64_{p.name}:{b64}")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
