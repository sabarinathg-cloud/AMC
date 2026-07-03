from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable

from .models import SegmentRecord

# How many rows to buffer before flushing a Parquet row group. Bounds peak memory to
# O(batch) rather than O(total rows): at millions of rows/shard the old "build the whole
# rows list + a pandas DataFrame" path spiked several GB on a 16 GB box.
_PARQUET_BATCH_SIZE = 50_000


def write_segment_manifests(
    output_root: Path,
    segments: Iterable[SegmentRecord],
    extra_rows: dict[str, dict[str, Any]] | None = None,
    enable_dataframe_exports: bool = True,
    xlsx_max_rows: int = 1_000_000,
    *,
    write_csv: bool = True,
    write_per_year: bool = True,
    parquet_batch_size: int = _PARQUET_BATCH_SIZE,
) -> list[Path]:
    """Stream per-segment manifests to disk, crash-safely and with bounded memory.

    Outputs (under ``<output_root>/manifests/``):
      * ``all_segments.jsonl``   -- ALWAYS. The durable, stream-readable record of the run
        (one JSON object per line). This is the safety-net "WAL": cheap to write, no parse.
      * ``all_segments.parquet`` -- if pyarrow is present. The analytics dataset, written in
        row-group batches via ``ParquetWriter`` so memory stays flat regardless of row count.
        Columns are written as Parquet strings (JSON fields are already strings; numeric
        fields are emitted as text and the readers -- ops/inspect_call.py, ops/merge_shards.sh
        -- already coerce). An all-string schema is what makes the streaming write
        schema-stable across batches and across shards.
      * ``all_segments.csv``     -- only if ``write_csv``. A single CSV is multi-GB and slow to
        parse at scale, so it is opt-out (the orchestrator disables it for large runs).
      * ``review.xlsx``          -- if ``enable_dataframe_exports`` and row count <= xlsx_max_rows.
        Streamed from the jsonl row-by-row via openpyxl write-only mode (constant memory, no
        pandas), so it never OOMs; skipped entirely on any error since jsonl/parquet are the
        lossless record. At scale the orchestrator disables it (``--manifest-no-xlsx``).
      * ``<year>/manifests/segments.{jsonl,csv}`` -- only if ``write_per_year``. Per-year
        duplicates of the global data; opt-out at scale (the merge step uses the global files).

    Crash-safety: every file is written to a temporary sibling and only ``os.replace``'d into
    place after it is fully written and closed, so a crash/kill/OOM never leaves a partial or
    corrupt manifest -- a reader (or ops/merge_shards.sh) only ever sees the previous complete
    file or the new complete file. Combined with the fact that this stage reads exclusively
    from the durable state store, a re-run after ANY interruption regenerates every file from
    scratch: no data loss, and no need to re-run earlier stages.
    """
    output_root = Path(output_root)
    extra_rows = extra_rows or {}
    global_dir = output_root / "manifests"
    global_dir.mkdir(parents=True, exist_ok=True)

    # A stable column union is required up front so the streaming Parquet/CSV writers keep a
    # constant schema across batches. base columns are fixed (SegmentRecord); extra columns are
    # the union of all per-segment extras (transcripts, pii, mask intervals, redacted paths...).
    fieldnames = _resolve_fieldnames(extra_rows)

    writers = _ManifestWriters(
        global_dir=global_dir,
        output_root=output_root,
        fieldnames=fieldnames,
        write_csv=write_csv,
        write_per_year=write_per_year,
        enable_parquet=enable_dataframe_exports,
        parquet_batch_size=parquet_batch_size,
    )
    try:
        for segment in segments:
            row = _segment_row(output_root, segment, extra_rows.get(segment.segment_id, {}))
            writers.add(row, str(getattr(segment, "year", row.get("year", "")) or ""))
        paths = writers.commit()
    except BaseException:
        writers.abort()
        raise

    # XLSX is only a human-review convenience for SMALL datasets; JSONL (durable) and Parquet
    # (analytics) are the real, lossless outputs. Generate it by STREAMING the jsonl we just
    # wrote row-by-row into an openpyxl write-only workbook (constant memory), and only when the
    # row count is within the cap. This can never OOM regardless of dataset size, and if it fails
    # for any reason it is simply skipped -- the jsonl/parquet already hold every row.
    if enable_dataframe_exports and 0 < writers.total <= xlsx_max_rows:
        xlsx = _write_xlsx_streaming(global_dir, fieldnames)
        if xlsx is not None:
            paths.append(xlsx)
    return paths


def _resolve_fieldnames(extra_rows: dict[str, dict[str, Any]]) -> list[str]:
    cols = _base_fieldnames()
    for extra in extra_rows.values():
        cols.update(extra.keys())
    return sorted(cols)


def _base_fieldnames() -> set[str]:
    names = {f.name for f in dataclass_fields(SegmentRecord)}
    # _segment_row drops the two Path fields and adds the abs/rel string forms.
    names.discard("source_path")
    names.discard("segment_audio_path")
    names.update({"source_path_abs", "segment_audio_path_abs", "segment_audio_path_rel"})
    return names


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


def _tmp_for(path: Path) -> Path:
    # Temp sibling in the SAME directory so os.replace() is an atomic rename on the same
    # filesystem (works on Lustre/NFS). The pid keeps concurrent writers from colliding.
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _to_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


class _ManifestWriters:
    """Owns every open temp file for a manifest write; commits atomically or cleans up on abort."""

    def __init__(
        self,
        *,
        global_dir: Path,
        output_root: Path,
        fieldnames: list[str],
        write_csv: bool,
        write_per_year: bool,
        enable_parquet: bool,
        parquet_batch_size: int,
    ) -> None:
        self.output_root = output_root
        self.fieldnames = fieldnames
        self.write_csv = write_csv
        self.write_per_year = write_per_year
        self.total = 0
        self._renames: list[tuple[Path, Path]] = []
        self._open_files: list[Any] = []

        # Global JSONL (always).
        self._jsonl_final = global_dir / "all_segments.jsonl"
        self._jsonl_tmp = _tmp_for(self._jsonl_final)
        self._jf = self._jsonl_tmp.open("w")
        self._open_files.append(self._jf)

        # Global CSV (optional).
        self._cw = None
        self._csv_final = self._csv_tmp = None
        if write_csv:
            self._csv_final = global_dir / "all_segments.csv"
            self._csv_tmp = _tmp_for(self._csv_final)
            cf = self._csv_tmp.open("w", newline="")
            self._open_files.append(cf)
            self._cw = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            self._cw.writeheader()

        # Global Parquet (optional, streamed).
        self._pq = _ParquetStream(global_dir / "all_segments.parquet", fieldnames, parquet_batch_size) if enable_parquet else None

        # Per-year writers, opened lazily as years are seen.
        self._year_writers: dict[str, _PerYearWriter] = {}

    def add(self, row: dict[str, Any], year: str) -> None:
        self.total += 1
        line = json.dumps(row, sort_keys=True, default=str)
        self._jf.write(line + "\n")
        if self._cw is not None:
            self._cw.writerow(row)
        if self._pq is not None:
            self._pq.add(row)
        if self.write_per_year and year:
            self._year_writer(year).add(row, line)

    def _year_writer(self, year: str) -> "_PerYearWriter":
        w = self._year_writers.get(year)
        if w is None:
            year_dir = self.output_root / year / "manifests"
            year_dir.mkdir(parents=True, exist_ok=True)
            w = _PerYearWriter(year_dir, self.fieldnames, self.write_csv)
            self._open_files.extend(w.open_files)
            self._year_writers[year] = w
        return w

    def commit(self) -> list[Path]:
        # Finish row groups + close all handles, then publish via atomic rename.
        if self._pq is not None:
            self._renames.append(self._pq.finish())
        for f in self._open_files:
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
            f.close()
        self._renames.append((self._jsonl_tmp, self._jsonl_final))
        if self._csv_tmp is not None and self._csv_final is not None:
            self._renames.append((self._csv_tmp, self._csv_final))
        for w in self._year_writers.values():
            self._renames.extend(w.renames())

        paths: list[Path] = []
        for tmp, final in self._renames:
            if tmp is None or final is None:
                continue
            os.replace(tmp, final)
            paths.append(final)
        return paths

    def abort(self) -> None:
        for f in self._open_files:
            try:
                f.close()
            except Exception:
                pass
        if self._pq is not None:
            self._pq.abort()
        # Best-effort cleanup of every temp file so a crash does not litter the tree.
        for tmp in [self._jsonl_tmp, self._csv_tmp]:
            _unlink(tmp)
        for w in self._year_writers.values():
            w.abort()


class _PerYearWriter:
    def __init__(self, year_dir: Path, fieldnames: list[str], write_csv: bool) -> None:
        self.open_files: list[Any] = []
        self._jsonl_final = year_dir / "segments.jsonl"
        self._jsonl_tmp = _tmp_for(self._jsonl_final)
        self._jf = self._jsonl_tmp.open("w")
        self.open_files.append(self._jf)
        self._cw = None
        self._csv_final = self._csv_tmp = None
        if write_csv:
            self._csv_final = year_dir / "segments.csv"
            self._csv_tmp = _tmp_for(self._csv_final)
            cf = self._csv_tmp.open("w", newline="")
            self.open_files.append(cf)
            self._cw = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            self._cw.writeheader()

    def add(self, row: dict[str, Any], line: str) -> None:
        self._jf.write(line + "\n")
        if self._cw is not None:
            self._cw.writerow(row)

    def renames(self) -> list[tuple[Path, Path]]:
        out = [(self._jsonl_tmp, self._jsonl_final)]
        if self._csv_tmp is not None and self._csv_final is not None:
            out.append((self._csv_tmp, self._csv_final))
        return out

    def abort(self) -> None:
        _unlink(self._jsonl_tmp)
        _unlink(self._csv_tmp)


class _ParquetStream:
    """Streams rows to a Parquet file one row group per batch, all columns as strings.

    All-string schema is deliberate: it stays constant across batches (and across shards), so
    the dataset merge in ops/merge_shards.sh and the readers can union the per-shard parts
    without dtype conflicts. JSON columns are already strings; numeric columns are emitted as
    text and the readers coerce.
    """

    def __init__(self, final_path: Path, fieldnames: list[str], batch_size: int) -> None:
        self.final = final_path
        self.tmp: Path | None = None
        self.fieldnames = fieldnames
        self.batch_size = max(1, batch_size)
        self._buf: list[dict[str, Any]] = []
        self._writer = None
        self._pa = None
        self._schema = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self._pa = pa
            self._schema = pa.schema([(name, pa.string()) for name in fieldnames])
            self.tmp = _tmp_for(final_path)
            try:
                self._writer = pq.ParquetWriter(str(self.tmp), self._schema, compression="zstd")
            except Exception:
                # zstd may be unavailable in a minimal pyarrow build; fall back to default codec.
                self._writer = pq.ParquetWriter(str(self.tmp), self._schema)
        except Exception:
            # pyarrow not installed -> skip parquet entirely (jsonl remains the durable export).
            self._writer = None
            self.tmp = None

    def add(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            return
        self._buf.append(row)
        if len(self._buf) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if self._writer is None or not self._buf:
            return
        cols = {name: [_to_cell(r.get(name)) for r in self._buf] for name in self.fieldnames}
        table = self._pa.table(cols, schema=self._schema)
        self._writer.write_table(table)
        self._buf.clear()

    def finish(self) -> tuple[Path | None, Path | None]:
        if self._writer is None:
            return (None, None)
        self._flush()
        self._writer.close()
        return (self.tmp, self.final)

    def abort(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception:
            pass
        _unlink(self.tmp)


# Excel hard limits: 1,048,576 rows/sheet and 32,767 chars/cell. We stay within both.
_XLSX_MAX_SHEET_ROWS = 1_048_576
_XLSX_MAX_CELL_CHARS = 32_767


def _xlsx_cell(value: Any) -> str | None:
    """Coerce a value to an Excel-safe cell: string, length-capped, control-chars stripped."""
    s = _to_cell(value)
    if s is None:
        return None
    if len(s) > _XLSX_MAX_CELL_CHARS:
        s = s[:_XLSX_MAX_CELL_CHARS]
    # openpyxl rejects illegal XML control chars; strip them so a stray byte can't abort the write.
    return "".join(ch for ch in s if ch >= " " or ch in "\t\n\r")


def _write_xlsx_streaming(global_dir: Path, fieldnames: list[str]) -> Path | None:
    """Write review.xlsx by streaming the durable jsonl row-by-row (constant memory).

    Uses openpyxl's write_only workbook, which flushes rows to a temp file as they are
    appended instead of holding the whole sheet in RAM -- so peak memory is O(1 row), not
    O(all rows). Any failure (openpyxl missing, bad cell, Excel row cap) is swallowed and the
    xlsx is skipped: the jsonl + parquet already contain every row, so nothing is ever lost.
    """
    jsonl = global_dir / "all_segments.jsonl"
    if not jsonl.exists():
        return None
    try:
        from openpyxl import Workbook
    except Exception:
        # openpyxl not installed -> skip xlsx entirely; jsonl (durable) + parquet remain.
        return None

    final = global_dir / "review.xlsx"
    tmp = _tmp_for(final)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("segments")
    try:
        ws.append(list(fieldnames))  # header
        written = 1
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if written >= _XLSX_MAX_SHEET_ROWS:
                    # Would exceed Excel's per-sheet row limit; stop cleanly. The full data is
                    # in jsonl/parquet -- xlsx is only ever a bounded human-review view.
                    break
                row = json.loads(line)
                ws.append([_xlsx_cell(row.get(name)) for name in fieldnames])
                written += 1
        wb.save(str(tmp))
        os.replace(tmp, final)
        return final
    except BaseException:
        try:
            wb.close()
        except Exception:
            pass
        _unlink(tmp)
        return None


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
