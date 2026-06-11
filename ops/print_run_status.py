#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print AMC sharded run status from shared storage.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_status(args.run_root)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    banner = preflight_banner(args.run_root)
    if banner:
        print(banner)
    if not rows:
        print(f"No shard status found under {args.run_root}")
        return 0
    headers = [
        "shard",
        "state",
        "stage",
        "instance",
        "files",
        "segments",
        "model_results",
        "consensus",
        "pii",
        "alignment",
        "redacted",
        "failures",
        "input_files",
        "input_signature",
        "updated_at",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return 0


def preflight_banner(run_root: Path) -> str:
    """One-line summary of the pre-stage gates (HSM prewarm, discovery build) so
    the table shows movement before any shard reaches a processing stage."""
    parts: list[str] = []
    prewarm = _read_json(run_root / "status" / "prewarm.json")
    if prewarm:
        state = prewarm.get("state", "?")
        rem = prewarm.get("released_remaining")
        total = prewarm.get("total_files")
        if state == "done":
            parts.append(f"prewarm: done ({total} files resident)")
        else:
            parts.append(f"prewarm: {state} (released remaining={rem}/{total})")
    for cache_dir in sorted((run_root / "discovery-cache").glob("*/")):
        if (cache_dir / ".done").exists():
            meta = _read_json(cache_dir / "meta.json")
            parts.append(f"discovery: done ({meta.get('total', '?')} files)")
        else:
            prog = _read_json(cache_dir / "progress.json")
            if prog:
                parts.append(f"discovery: building ({prog.get('seen', 0)} files seen)")
            else:
                parts.append("discovery: starting")
        break
    return " | ".join(parts)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect_status(run_root: Path) -> list[dict[str, Any]]:
    status_dir = run_root / "status"
    outputs_dir = run_root / "outputs"
    status_by_shard: dict[str, dict[str, Any]] = {}
    for path in sorted(status_dir.glob("shard-*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            payload = {"state": "unreadable", "message": repr(exc)}
        shard = path.stem
        status_by_shard[shard] = payload

    shards = sorted(set(status_by_shard) | {p.name for p in outputs_dir.glob("shard-*") if p.is_dir()})
    rows = []
    for shard in shards:
        status = status_by_shard.get(shard, {})
        out_dir = outputs_dir / shard
        counts = sqlite_counts(out_dir / ".pii_pipeline" / "state" / "pipeline.sqlite3")
        rows.append(
            {
                "shard": shard.replace("shard-", ""),
                "state": status.get("state", ""),
                "stage": status.get("stage", ""),
                "instance": status.get("instance_id", ""),
                "files": counts.get("files", ""),
                "segments": counts.get("segments", ""),
                "model_results": counts.get("model_results", ""),
                "consensus": counts.get("consensus", ""),
                "pii": counts.get("pii", ""),
                "alignment": counts.get("alignment", ""),
                "redacted": counts.get("redacted", ""),
                "failures": counts.get("failures", ""),
                "input_files": status.get("input_files", ""),
                "input_signature": status.get("input_signature", ""),
                "updated_at": status.get("updated_at", ""),
                "message": status.get("message", ""),
            }
        )
    return rows


def sqlite_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with sqlite3.connect(path) as conn:
        counts = {
            "files": _count(conn, "files"),
            "segments": _count(conn, "segments"),
            "model_results": _count(conn, "model_results"),
            "failures": _count(conn, "failures"),
            "consensus": _artifact_count(conn, "consensus"),
            "pii": _artifact_count(conn, "pii"),
            "alignment": _artifact_count(conn, "alignment"),
            "redacted": _artifact_count(conn, "redacted"),
        }
    return counts


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"select count(*) from {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _artifact_count(conn: sqlite3.Connection, kind: str) -> int:
    try:
        return int(conn.execute("select count(*) from artifacts where kind = ?", (kind,)).fetchone()[0])
    except sqlite3.Error:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
