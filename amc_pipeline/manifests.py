from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import SegmentRecord


def write_segment_manifests(output_root: Path, segments: list[SegmentRecord], extra_rows: dict[str, dict[str, Any]] | None = None) -> list[Path]:
    output_root = Path(output_root)
    paths: list[Path] = []
    rows = [_segment_row(output_root, s, extra_rows.get(s.segment_id, {}) if extra_rows else {}) for s in segments]
    global_dir = output_root / "manifests"
    global_dir.mkdir(parents=True, exist_ok=True)
    paths.append(_write_csv(global_dir / "all_segments.csv", rows))
    paths.append(_write_jsonl(global_dir / "all_segments.jsonl", rows))
    for year in sorted({s.year for s in segments}):
        year_rows = [r for r in rows if r["year"] == year]
        year_dir = output_root / year / "manifests"
        year_dir.mkdir(parents=True, exist_ok=True)
        paths.append(_write_csv(year_dir / "segments.csv", year_rows))
        paths.append(_write_jsonl(year_dir / "segments.jsonl", year_rows))
    _try_optional_dataframe_exports(global_dir, rows)
    return paths


def _segment_row(output_root: Path, segment: SegmentRecord, extra: dict[str, Any]) -> dict[str, Any]:
    row = asdict(segment)
    row["source_path_abs"] = str(segment.source_path)
    row["segment_audio_path_abs"] = str(segment.segment_audio_path)
    row["segment_audio_path_rel"] = _safe_relpath(segment.segment_audio_path, output_root)
    row.pop("source_path", None)
    row.pop("segment_audio_path", None)
    row.update(extra)
    return row


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def _try_optional_dataframe_exports(global_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        try:
            df.to_parquet(global_dir / "all_segments.parquet", index=False)
        except Exception:
            pass
        try:
            df.to_excel(global_dir / "review.xlsx", index=False)
        except Exception:
            pass
    except Exception:
        return
