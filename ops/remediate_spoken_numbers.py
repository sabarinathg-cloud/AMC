#!/usr/bin/env python3
"""Invalidate exactly the work that the spoken-number PII fix changes, for one shard.

32 shards finished their `pii` stage before `spoken_numbers` existed, so every phone
number they heard read out digit by digit went unmasked. Those shards need pii ->
manifest re-run -- but re-running them wholesale would cost ~4h each, and almost all
of that is `align`, which is only wrong for the ~0.5% of segments whose spans changed.

Because the new detector is pure text, the affected segments can be identified offline
from the manifest without running a single model. This deletes the per-item artifacts
for just those segments, which is what makes the pipeline redo them: every stage skips
items that already have an artifact, so deleting one artifact re-does one segment.

    python3 ops/remediate_spoken_numbers.py --run RUN_ROOT --shard 0            # inspect
    python3 ops/remediate_spoken_numbers.py --run RUN_ROOT --shard 0 --apply    # invalidate

Then re-run the stages (skips everything still marked done):

    STAGES="pii align mask_plan redact validate manifest" SHARD_INDEX=0 \
      bash ops/run_shard_no_docker.sh
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amc_pipeline.spoken_numbers import find_spoken_number_runs  # noqa: E402

# Deleting a stage's marker is what lets the stage run again at all; deleting an item's
# artifact is what stops it from skipping that item. Both are needed.
REDO_STAGES = ["pii", "align", "mask_plan", "redact", "validate", "manifest"]


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=120.0)
    con.execute("pragma busy_timeout=120000")
    return con


def affected_segments(manifest: Path) -> tuple[set[str], int]:
    """Segment ids whose final transcript contains a spoken number, and total rows."""
    import pyarrow.parquet as pq

    table = pq.read_table(manifest, columns=["segment_id", "final_transcript"]).to_pydict()
    hits = {
        seg
        for seg, text in zip(table["segment_id"], table["final_transcript"])
        if find_spoken_number_runs(text or "")
    }
    return hits, len(table["segment_id"])


def file_ids_for(con: sqlite3.Connection, segments: set[str]) -> set[str]:
    out: set[str] = set()
    rows = con.execute("select segment_id, file_id from segments").fetchall()
    for segment_id, file_id in rows:
        if segment_id in segments and file_id:
            out.add(file_id)
    return out


def delete_artifacts(con: sqlite3.Connection, ids: list[str]) -> int:
    total = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" * len(chunk))
        cur = con.execute(f"delete from artifacts where artifact_id in ({marks})", chunk)
        total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    return total


def verify(manifest: Path) -> int:
    """After remediation, every spoken number must carry a mask. Returns leak count."""
    import pyarrow.parquet as pq

    cols = ["segment_id", "final_transcript", "pii_count", "mask_intervals_json"]
    man = pq.read_table(manifest, columns=cols).to_pydict()
    total = with_number = still_open = 0
    examples: list[str] = []
    for i in range(len(man["segment_id"])):
        total += 1
        text = man["final_transcript"][i] or ""
        if not find_spoken_number_runs(text):
            continue
        with_number += 1
        try:
            masked = bool(json.loads(man["mask_intervals_json"][i] or "[]"))
        except (json.JSONDecodeError, TypeError):
            masked = False
        if not masked:
            still_open += 1
            if len(examples) < 8:
                examples.append(" ".join(text.split())[:100])
    print(f"  {total:,} segments, {with_number:,} contain a spoken number, "
          f"{with_number - still_open:,} now masked, {still_open:,} still unmasked")
    for line in examples:
        print(f"    UNMASKED: {line}")
    return still_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run root, e.g. .../2025-full")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: inspect)")
    ap.add_argument("--verify", action="store_true",
                    help="check a remediated shard instead: exits 1 if any number is still bare")
    args = ap.parse_args()

    if args.verify:
        manifest = args.run / "outputs" / f"shard-{args.shard}" / "manifests" / "all_segments.parquet"
        if not manifest.exists():
            print(f"missing: {manifest}", file=sys.stderr)
            return 2
        print(f"shard-{args.shard}: verifying")
        return 1 if verify(manifest) else 0

    shard_dir = args.run / "outputs" / f"shard-{args.shard}"
    db = shard_dir / ".pii_pipeline" / "state" / "pipeline.sqlite3"
    manifest = shard_dir / "manifests" / "all_segments.parquet"
    markers = shard_dir / ".pii_pipeline" / "stage_markers"
    for path in (db, manifest):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    segments, total = affected_segments(manifest)
    con = connect(db)
    try:
        files = file_ids_for(con, segments)
        print(f"shard-{args.shard}: {total:,} segments, {len(segments):,} with a spoken number "
              f"({len(segments)/max(total,1)*100:.2f}%), across {len(files):,} calls")

        artifact_ids = (
            [f"pii:{s}" for s in sorted(segments)]
            + [f"alignment:{s}" for s in sorted(segments)]
            + [f"mask_plan:{f}" for f in sorted(files)]
            + [f"redacted:{f}" for f in sorted(files)]
            + [f"validation:{f}" for f in sorted(files)]
        )
        present = 0
        for i in range(0, len(artifact_ids), 500):
            chunk = artifact_ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            present += con.execute(
                f"select count(*) from artifacts where artifact_id in ({marks})", chunk
            ).fetchone()[0]
        print(f"  artifacts to invalidate: {present:,} of {len(artifact_ids):,} candidate ids")
        stale = [m for m in REDO_STAGES if (markers / f"{m}.done").exists()]
        print(f"  stage markers to clear: {', '.join(stale) or '(none)'}")

        if not segments:
            print("  nothing to do")
            return 0
        if not args.apply:
            print("\ninspect only -- pass --apply to invalidate")
            return 0

        deleted = delete_artifacts(con, artifact_ids)
        con.commit()
        for marker in stale:
            (markers / f"{marker}.done").unlink(missing_ok=True)
        # The run's status file is what watch_run.sh reads; leave a breadcrumb so a
        # shard that is being re-run is not mistaken for one that never finished.
        note = shard_dir / ".pii_pipeline" / "reports" / "spoken_number_remediation.json"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(json.dumps({
            "shard": args.shard,
            "segments_invalidated": len(segments),
            "calls_invalidated": len(files),
            "artifacts_deleted": deleted,
            "stages_cleared": stale,
        }, indent=2, sort_keys=True))
        print(f"  deleted {deleted:,} artifacts; cleared {len(stale)} markers")
        print(f"\nnow re-run:  STAGES=\"{' '.join(REDO_STAGES)}\" SHARD_INDEX={args.shard} "
              f"bash ops/run_shard_no_docker.sh")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
