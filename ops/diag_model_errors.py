#!/usr/bin/env python3
"""Per-model ASR failure breakdown across a run's shard DBs.

For each ASR model it reports how many segment results are transcribed vs failed,
and for the failures it dedupes the stored error strings with counts -- so you see
the *actual exception*, not just an empty transcript. Stdlib only (any venv).

Examples
--------
    python3 ops/diag_model_errors.py --run-root /mnt/amc-data/amc-runs/2026-smoke-50
    python3 ops/diag_model_errors.py --run-root /mnt/amc-data/amc-runs/2026-smoke-50 --model cohere
    python3 ops/diag_model_errors.py --db /path/to/pipeline.sqlite3
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path


def _candidate_dbs(args) -> list[str]:
    if args.db:
        return [args.db]
    if args.out:
        return [str(Path(args.out) / ".pii_pipeline" / "state" / "pipeline.sqlite3")]
    root = args.run_root or os.environ.get("AMC_RUNS_ROOT") or "/mnt/amc-data/amc-runs"
    suffix = os.path.join("outputs", "shard-*", ".pii_pipeline", "state", "pipeline.sqlite3")
    dbs: list[str] = []
    seen: set[str] = set()
    for pat in (os.path.join(root, suffix), os.path.join(root, "*", suffix), os.path.join(root, "*", "*", suffix)):
        for db in glob.glob(pat):
            if db not in seen and os.path.exists(db):
                seen.add(db)
                dbs.append(db)
    return sorted(dbs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-model ASR failure breakdown across shard DBs")
    ap.add_argument("--db")
    ap.add_argument("--out")
    ap.add_argument("--run-root")
    ap.add_argument("--model", help="restrict to one model (whisper/qwen/cohere/granite)")
    ap.add_argument("--top", type=int, default=10, help="show this many distinct error strings per model")
    ap.add_argument("--maxlen", type=int, default=400, help="truncate each error string to this length")
    args = ap.parse_args()

    dbs = _candidate_dbs(args)
    if not dbs:
        print("FATAL: no state DB found. Pass --db / --out / --run-root.", file=sys.stderr)
        return 2

    # model -> {"transcribed": n, "failed": n}; model -> Counter(error_string)
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    errors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    samples: dict[str, str] = {}  # model -> a sample failing segment_id

    for db in dbs:
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except Exception as exc:
            print(f"WARN: cannot open {db}: {exc}", file=sys.stderr)
            continue
        sql = "select segment_id, model_name, status, payload_json from model_results"
        params: tuple = ()
        if args.model:
            sql += " where model_name = ?"
            params = (args.model,)
        try:
            rows = con.execute(sql, params).fetchall()
        except Exception as exc:
            print(f"WARN: query failed on {db}: {exc}", file=sys.stderr)
            con.close()
            continue
        con.close()
        for segment_id, model_name, status, pj in rows:
            counts[model_name][status] += 1
            if status == "failed":
                try:
                    payload = json.loads(pj or "{}")
                except Exception:
                    payload = {}
                err = str(payload.get("error") or "<empty>")[: args.maxlen]
                errors[model_name][err] += 1
                samples.setdefault(model_name + "|" + err, segment_id)

    print(f"scanned {len(dbs)} shard DB(s) under {args.run_root or args.out or args.db}")
    for model in sorted(counts):
        c = counts[model]
        total = sum(c.values())
        line = "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print("=" * 72)
        print(f"MODEL {model}: total={total}  {line}")
        if errors[model]:
            for err, n in errors[model].most_common(args.top):
                sid = samples.get(model + "|" + err, "")
                print(f"  ---- {n}x  (e.g. {sid}) ----")
                print(f"  {err}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
