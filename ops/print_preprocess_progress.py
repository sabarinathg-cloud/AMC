#!/usr/bin/env python3
"""Per-shard preprocess progress table.

Reads every shard DB under <run-root>/outputs/shard-*/.pii_pipeline/state and
prints the file status breakdown (preprocessed / preprocessing / other) plus
segment count, so progress is visible even though the run_status `files` column
(a row COUNT) stays flat while rows are UPDATED in place.

All shard DBs sit on the shared mount, so this runs from any single instance.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()

    shard_dirs = sorted((args.run_root / "outputs").glob("shard-*"))
    rows = []
    statuses: set[str] = set()
    for d in shard_dirs:
        db = d / ".pii_pipeline" / "state" / "pipeline.sqlite3"
        rec = {"shard": d.name.replace("shard-", ""), "_by": {}, "segments": 0, "total": 0}
        if db.exists():
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                    rec["_by"] = dict(c.execute("select status,count(*) from files group by status").fetchall())
                    rec["segments"] = c.execute("select count(*) from segments").fetchone()[0]
                    rec["total"] = c.execute("select count(*) from files").fetchone()[0]
            except sqlite3.Error as exc:
                rec["_by"] = {"error": str(exc)}
        statuses.update(k for k in rec["_by"] if k != "error")
        rows.append(rec)

    # Stable column order: preprocessed/preprocessing first, then the rest.
    ordered = [s for s in ("preprocessed", "preprocessing", "inspected") if s in statuses]
    ordered += sorted(s for s in statuses if s not in ordered)
    headers = ["shard", *ordered, "segments", "total_files"]

    table = []
    for rec in rows:
        row = {"shard": rec["shard"], "segments": rec["segments"], "total_files": rec["total"]}
        for s in ordered:
            row[s] = rec["_by"].get(s, 0)
        table.append(row)

    widths = {h: len(h) for h in headers}
    for row in table:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in table:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
