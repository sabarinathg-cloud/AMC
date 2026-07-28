from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


_FETCH_BATCH_SIZE = 1000


class SQLiteStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("select 1")
            except sqlite3.ProgrammingError:
                self._local.conn = None
                conn = None
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("pragma journal_mode=WAL")
            conn.execute("pragma synchronous=NORMAL")
            conn.execute("pragma busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _read_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout=30000")
        return conn

    @contextmanager
    def _conn_ctx(self):
        yield self.connect()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except sqlite3.Error:
            pass
        finally:
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        with closing(self._read_connection()) as conn:
            conn.executescript(
                """
                create table if not exists files (
                    file_id text primary key,
                    source_path text unique not null,
                    status text not null,
                    payload_json text not null,
                    updated_at real not null
                );
                create table if not exists segments (
                    segment_id text primary key,
                    file_id text not null,
                    status text not null,
                    payload_json text not null,
                    updated_at real not null
                );
                create table if not exists model_results (
                    segment_id text not null,
                    model_name text not null,
                    status text not null,
                    payload_json text not null,
                    updated_at real not null,
                    primary key (segment_id, model_name)
                );
                create table if not exists artifacts (
                    artifact_id text primary key,
                    kind text not null,
                    path text not null,
                    status text not null,
                    payload_json text not null,
                    updated_at real not null
                );
                create table if not exists failures (
                    failure_id text primary key,
                    scope text not null,
                    scope_id text not null,
                    error text not null,
                    traceback text,
                    retryable integer not null,
                    created_at real not null
                );
                create table if not exists leases (
                    lease_id text primary key,
                    worker_id text not null,
                    stage text not null,
                    scope_id text not null,
                    expires_at real not null,
                    payload_json text not null
                );
                create table if not exists pause_requests (
                    pause_id integer primary key autoincrement,
                    run_id text not null,
                    worker_id text,
                    global_pause integer not null,
                    created_at real not null,
                    released_at real
                );
                create table if not exists run_metadata (
                    key text primary key,
                    value_json text not null,
                    updated_at real not null
                );
                """
            )

    def upsert_file(self, record: dict[str, Any]) -> None:
        payload = dict(record.get("payload") or record)
        file_id = str(record["file_id"])
        source_path = str(record["source_path"])
        status = str(record.get("status", payload.get("status", "unknown")))
        now = time.time()
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert into files(file_id, source_path, status, payload_json, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(file_id) do update set
                    source_path=excluded.source_path,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (file_id, source_path, status, json.dumps(payload, sort_keys=True, default=str), now),
            )

    def upsert_segment(self, segment_id: str, file_id: str, status: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert into segments(segment_id, file_id, status, payload_json, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(segment_id) do update set
                    file_id=excluded.file_id,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (segment_id, file_id, status, json.dumps(payload, sort_keys=True, default=str), now),
            )

    def upsert_segments_many(self, segments: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """Upsert many segments in a SINGLE transaction.

        persist() previously issued one autocommit transaction per segment
        (~12-20 per file). On a Lustre-backed WAL DB that transaction/sync count
        dominated preprocess wall-time. Wrapping a whole file's segments in one
        BEGIN/COMMIT cuts it ~15x. ``segments`` is ``(segment_id, file_id, status, payload)``.
        """
        if not segments:
            return
        now = time.time()
        rows = [
            (str(sid), str(fid), str(st), json.dumps(pl, sort_keys=True, default=str), now)
            for (sid, fid, st, pl) in segments
        ]
        with self._conn_ctx() as conn:
            try:
                conn.execute("begin")
                conn.executemany(
                    """
                    insert into segments(segment_id, file_id, status, payload_json, updated_at)
                    values (?, ?, ?, ?, ?)
                    on conflict(segment_id) do update set
                        file_id=excluded.file_id,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )
                conn.execute("commit")
            except Exception:
                try:
                    conn.execute("rollback")
                except Exception:
                    pass
                raise

    def upsert_model_result(self, segment_id: str, model_name: str, status: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert into model_results(segment_id, model_name, status, payload_json, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(segment_id, model_name) do update set
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (segment_id, model_name, status, json.dumps(payload, sort_keys=True, default=str), now),
            )

    def upsert_model_results_many(self, results: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """Upsert many model results in a SINGLE transaction.

        Mirrors upsert_segments_many: lets the ASR stage persist a chunk of
        transcripts at once (live progress + mid-stage resumability) without one
        autocommit transaction per segment on the Lustre-backed WAL DB.
        ``results`` is ``(segment_id, model_name, status, payload)``.
        """
        if not results:
            return
        now = time.time()
        rows = [
            (str(sid), str(model), str(st), json.dumps(pl, sort_keys=True, default=str), now)
            for (sid, model, st, pl) in results
        ]
        with self._conn_ctx() as conn:
            try:
                conn.execute("begin")
                conn.executemany(
                    """
                    insert into model_results(segment_id, model_name, status, payload_json, updated_at)
                    values (?, ?, ?, ?, ?)
                    on conflict(segment_id, model_name) do update set
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )
                conn.execute("commit")
            except Exception:
                try:
                    conn.execute("rollback")
                except Exception:
                    pass
                raise

    def record_artifact(self, artifact_id: str, kind: str, path: Path, status: str, payload: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert into artifacts(artifact_id, kind, path, status, payload_json, updated_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(artifact_id) do update set
                    kind=excluded.kind,
                    path=excluded.path,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (artifact_id, kind, str(path), status, json.dumps(payload or {}, sort_keys=True, default=str), now),
            )

    def record_artifacts_many(self, artifacts: list[tuple[str, str, Path, str, dict[str, Any] | None]]) -> None:
        """Record many artifacts in a SINGLE transaction.

        Counterpart to upsert_model_results_many for stages whose per-item output is a
        cheap artifact row: most segments carry no PII spans, so the align stage writes
        an empty alignment for them and nothing else, and one autocommit transaction per
        empty row was the bulk of that stage's cost.
        ``artifacts`` is ``(artifact_id, kind, path, status, payload)``.
        """
        if not artifacts:
            return
        now = time.time()
        rows = [
            (str(aid), str(kind), str(path), str(status), json.dumps(payload or {}, sort_keys=True, default=str), now)
            for (aid, kind, path, status, payload) in artifacts
        ]
        with self._conn_ctx() as conn:
            try:
                conn.execute("begin")
                conn.executemany(
                    """
                    insert into artifacts(artifact_id, kind, path, status, payload_json, updated_at)
                    values (?, ?, ?, ?, ?, ?)
                    on conflict(artifact_id) do update set
                        kind=excluded.kind,
                        path=excluded.path,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )
                conn.execute("commit")
            except Exception:
                try:
                    conn.execute("rollback")
                except Exception:
                    pass
                raise

    def record_failure(self, failure_id: str, scope: str, scope_id: str, error: str, retryable: bool = True, traceback: str | None = None) -> None:
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert or replace into failures(failure_id, scope, scope_id, error, traceback, retryable, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (failure_id, scope, scope_id, error, traceback, int(retryable), time.time()),
            )

    def clear_failure(self, failure_id: str) -> None:
        """Delete a single failure row once its scope has succeeded on retry."""
        self.clear_failures([failure_id])

    def clear_failures(self, failure_ids: Iterable[str]) -> None:
        """Delete resolved failure rows so the `failures` count reflects only work
        that is still broken. No-op for ids that were never recorded as failures."""
        ids = [fid for fid in failure_ids if fid]
        if not ids:
            return
        with self._conn_ctx() as conn:
            conn.executemany("delete from failures where failure_id = ?", [(fid,) for fid in ids])

    def request_pause(self, run_id: str, worker_id: str | None = None, global_pause: bool = False) -> None:
        with self._conn_ctx() as conn:
            conn.execute(
                "insert into pause_requests(run_id, worker_id, global_pause, created_at) values (?, ?, ?, ?)",
                (run_id, worker_id, int(global_pause), time.time()),
            )

    def should_pause(self, run_id: str, worker_id: str | None = None) -> bool:
        with self._conn_ctx() as conn:
            row = conn.execute(
                """
                select 1 from pause_requests
                where run_id = ?
                  and released_at is null
                  and (global_pause = 1 or worker_id = ?)
                limit 1
                """,
                (run_id, worker_id),
            ).fetchone()
        return row is not None

    def set_run_metadata(self, key: str, value: Any) -> None:
        with self._conn_ctx() as conn:
            conn.execute(
                """
                insert into run_metadata(key, value_json, updated_at) values (?, ?, ?)
                on conflict(key) do update set value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, sort_keys=True, default=str), time.time()),
            )

    def table_count(self, table: str) -> int:
        if table not in {"files", "segments", "model_results", "artifacts", "failures", "leases", "pause_requests", "run_metadata"}:
            raise ValueError(f"Unsupported table: {table}")
        with self._conn_ctx() as conn:
            return int(conn.execute(f"select count(*) from {table}").fetchone()[0])

    def fetch_files(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from files"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = ?"
            params = (status,)
        with self._conn_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_segments(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from segments"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = ?"
            params = (status,)
        with self._conn_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_model_results(self, segment_id: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from model_results"
        params: tuple[Any, ...] = ()
        if segment_id is not None:
            sql += " where segment_id = ?"
            params = (segment_id,)
        with self._conn_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_artifacts(self, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from artifacts"
        params: tuple[Any, ...] = ()
        if kind is not None:
            sql += " where kind = ?"
            params = (kind,)
        with self._conn_ctx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def count_files(self, status: str | None = None) -> int:
        return self._count("files", "status", status)

    def count_segments(self, status: str | None = None) -> int:
        return self._count("segments", "status", status)

    def count_model_results(self, segment_id: str | None = None) -> int:
        return self._count("model_results", "segment_id", segment_id)

    def count_artifacts(self, kind: str | None = None) -> int:
        return self._count("artifacts", "kind", kind)

    def iter_files(self, status: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("files", "status", status, batch_size)

    def iter_segments(self, status: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("segments", "status", status, batch_size)

    def iter_model_results(self, segment_id: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("model_results", "segment_id", segment_id, batch_size)

    def iter_artifacts(self, kind: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("artifacts", "kind", kind, batch_size)

    def _count(self, table: str, filter_column: str, filter_value: str | None) -> int:
        sql = f"select count(*) from {table}"
        params: tuple[Any, ...] = ()
        if filter_value is not None:
            sql += f" where {filter_column} = ?"
            params = (filter_value,)
        with self._conn_ctx() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def _iter_rows(self, table: str, filter_column: str, filter_value: str | None, batch_size: int) -> Iterator[dict[str, Any]]:
        sql = f"select * from {table}"
        params: tuple[Any, ...] = ()
        if filter_value is not None:
            sql += f" where {filter_column} = ?"
            params = (filter_value,)
        conn = self._read_connection()
        try:
            cursor = conn.execute(sql, params)
            while True:
                rows = cursor.fetchmany(max(1, batch_size))
                if not rows:
                    break
                for row in rows:
                    yield self._decode_row(row)
        finally:
            conn.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ["payload_json", "value_json"]:
            if key in data:
                data["payload" if key == "payload_json" else "value"] = json.loads(data[key])
        return data


class PostgresStateStore:
    """Optional distributed state store.

    This class is intentionally dependency-light at import time. It imports
    psycopg only when a Postgres backend is requested.
    """

    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("Postgres backend requires a DSN")
        try:
            import psycopg  # type: ignore
        except Exception as exc:
            raise RuntimeError("Postgres backend requires the 'psycopg' package") from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self._init_schema()

    def connect(self):
        return self._psycopg.connect(self.dsn)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    create table if not exists files (
                        file_id text primary key,
                        source_path text unique not null,
                        status text not null,
                        payload_json jsonb not null,
                        updated_at double precision not null
                    );
                    create table if not exists segments (
                        segment_id text primary key,
                        file_id text not null,
                        status text not null,
                        payload_json jsonb not null,
                        updated_at double precision not null
                    );
                    create table if not exists model_results (
                        segment_id text not null,
                        model_name text not null,
                        status text not null,
                        payload_json jsonb not null,
                        updated_at double precision not null,
                        primary key (segment_id, model_name)
                    );
                    create table if not exists artifacts (
                        artifact_id text primary key,
                        kind text not null,
                        path text not null,
                        status text not null,
                        payload_json jsonb not null,
                        updated_at double precision not null
                    );
                    create table if not exists failures (
                        failure_id text primary key,
                        scope text not null,
                        scope_id text not null,
                        error text not null,
                        traceback text,
                        retryable integer not null,
                        created_at double precision not null
                    );
                    create table if not exists pause_requests (
                        pause_id bigserial primary key,
                        run_id text not null,
                        worker_id text,
                        global_pause integer not null,
                        created_at double precision not null,
                        released_at double precision
                    );
                    create table if not exists leases (
                        lease_id text primary key,
                        worker_id text not null,
                        stage text not null,
                        scope_id text not null,
                        expires_at double precision not null,
                        payload_json jsonb not null
                    );
                    create table if not exists run_metadata (
                        key text primary key,
                        value_json jsonb not null,
                        updated_at double precision not null
                    );
                    """
                )

    def upsert_file(self, record: dict[str, Any]) -> None:
        payload = dict(record.get("payload") or record)
        now = time.time()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into files(file_id, source_path, status, payload_json, updated_at)
                    values (%s, %s, %s, %s::jsonb, %s)
                    on conflict(file_id) do update set
                        source_path=excluded.source_path,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (str(record["file_id"]), str(record["source_path"]), str(record.get("status", payload.get("status", "unknown"))), json.dumps(payload, sort_keys=True, default=str), now),
                )

    def upsert_segment(self, segment_id: str, file_id: str, status: str, payload: dict[str, Any]) -> None:
        self._upsert_payload("segments", "segment_id", segment_id, {"file_id": file_id, "status": status}, payload)

    def upsert_segments_many(self, segments: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        if not segments:
            return
        now = time.time()
        rows = [
            (str(sid), str(fid), str(st), json.dumps(pl, sort_keys=True, default=str), now)
            for (sid, fid, st, pl) in segments
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into segments(segment_id, file_id, status, payload_json, updated_at)
                    values (%s, %s, %s, %s::jsonb, %s)
                    on conflict(segment_id) do update set
                        file_id=excluded.file_id,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )

    def upsert_model_result(self, segment_id: str, model_name: str, status: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into model_results(segment_id, model_name, status, payload_json, updated_at)
                    values (%s, %s, %s, %s::jsonb, %s)
                    on conflict(segment_id, model_name) do update set
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (segment_id, model_name, status, json.dumps(payload, sort_keys=True, default=str), now),
                )

    def upsert_model_results_many(self, results: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        if not results:
            return
        now = time.time()
        rows = [
            (str(sid), str(model), str(st), json.dumps(pl, sort_keys=True, default=str), now)
            for (sid, model, st, pl) in results
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into model_results(segment_id, model_name, status, payload_json, updated_at)
                    values (%s, %s, %s, %s::jsonb, %s)
                    on conflict(segment_id, model_name) do update set
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )

    def record_artifact(self, artifact_id: str, kind: str, path: Path, status: str, payload: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into artifacts(artifact_id, kind, path, status, payload_json, updated_at)
                    values (%s, %s, %s, %s, %s::jsonb, %s)
                    on conflict(artifact_id) do update set
                        kind=excluded.kind,
                        path=excluded.path,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (artifact_id, kind, str(path), status, json.dumps(payload or {}, sort_keys=True, default=str), now),
                )

    def record_artifacts_many(self, artifacts: list[tuple[str, str, Path, str, dict[str, Any] | None]]) -> None:
        if not artifacts:
            return
        now = time.time()
        rows = [
            (str(aid), str(kind), str(path), str(status), json.dumps(payload or {}, sort_keys=True, default=str), now)
            for (aid, kind, path, status, payload) in artifacts
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into artifacts(artifact_id, kind, path, status, payload_json, updated_at)
                    values (%s, %s, %s, %s, %s::jsonb, %s)
                    on conflict(artifact_id) do update set
                        kind=excluded.kind,
                        path=excluded.path,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    rows,
                )

    def record_failure(self, failure_id: str, scope: str, scope_id: str, error: str, retryable: bool = True, traceback: str | None = None) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into failures(failure_id, scope, scope_id, error, traceback, retryable, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    on conflict(failure_id) do update set
                        error=excluded.error,
                        traceback=excluded.traceback,
                        retryable=excluded.retryable,
                        created_at=excluded.created_at
                    """,
                    (failure_id, scope, scope_id, error, traceback, int(retryable), time.time()),
                )

    def clear_failure(self, failure_id: str) -> None:
        """Delete a single failure row once its scope has succeeded on retry."""
        self.clear_failures([failure_id])

    def clear_failures(self, failure_ids: Iterable[str]) -> None:
        """Delete resolved failure rows so the `failures` count reflects only work
        that is still broken. No-op for ids that were never recorded as failures."""
        ids = [fid for fid in failure_ids if fid]
        if not ids:
            return
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("delete from failures where failure_id = any(%s)", (ids,))

    def request_pause(self, run_id: str, worker_id: str | None = None, global_pause: bool = False) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into pause_requests(run_id, worker_id, global_pause, created_at) values (%s, %s, %s, %s)",
                    (run_id, worker_id, int(global_pause), time.time()),
                )

    def should_pause(self, run_id: str, worker_id: str | None = None) -> bool:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select 1 from pause_requests
                    where run_id = %s
                      and released_at is null
                      and (global_pause = 1 or worker_id = %s)
                    limit 1
                    """,
                    (run_id, worker_id),
                )
                return cur.fetchone() is not None

    def fetch_files(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._fetch_payload_table("files", status)

    def fetch_segments(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._fetch_payload_table("segments", status)

    def fetch_model_results(self, segment_id: str | None = None) -> list[dict[str, Any]]:
        sql = "select segment_id, model_name, status, payload_json from model_results"
        params: tuple[Any, ...] = ()
        if segment_id is not None:
            sql += " where segment_id = %s"
            params = (segment_id,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [{"segment_id": r[0], "model_name": r[1], "status": r[2], "payload": r[3]} for r in rows]

    def fetch_artifacts(self, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "select artifact_id, kind, path, status, payload_json from artifacts"
        params: tuple[Any, ...] = ()
        if kind is not None:
            sql += " where kind = %s"
            params = (kind,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [{"artifact_id": r[0], "kind": r[1], "path": r[2], "status": r[3], "payload": r[4]} for r in rows]

    def set_run_metadata(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into run_metadata(key, value_json, updated_at) values (%s, %s::jsonb, %s)
                    on conflict(key) do update set value_json=excluded.value_json, updated_at=excluded.updated_at
                    """,
                    (key, json.dumps(value, sort_keys=True, default=str), time.time()),
                )

    def table_count(self, table: str) -> int:
        if table not in {"files", "segments", "model_results", "artifacts", "failures", "leases", "pause_requests", "run_metadata"}:
            raise ValueError(f"Unsupported table: {table}")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"select count(*) from {table}")
                return int(cur.fetchone()[0])

    def count_files(self, status: str | None = None) -> int:
        return self._count("files", "status", status)

    def count_segments(self, status: str | None = None) -> int:
        return self._count("segments", "status", status)

    def count_model_results(self, segment_id: str | None = None) -> int:
        return self._count("model_results", "segment_id", segment_id)

    def count_artifacts(self, kind: str | None = None) -> int:
        return self._count("artifacts", "kind", kind)

    def _count(self, table: str, filter_column: str, filter_value: str | None) -> int:
        sql = f"select count(*) from {table}"
        params: tuple[Any, ...] = ()
        if filter_value is not None:
            sql += f" where {filter_column} = %s"
            params = (filter_value,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return int(cur.fetchone()[0])

    def iter_files(self, status: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_payload_table("files", "status", status, batch_size)

    def iter_segments(self, status: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        yield from self._iter_payload_table("segments", "status", status, batch_size)

    def iter_model_results(self, segment_id: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        sql = "select segment_id, model_name, status, payload_json from model_results"
        params: tuple[Any, ...] = ()
        if segment_id is not None:
            sql += " where segment_id = %s"
            params = (segment_id,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                while True:
                    rows = cur.fetchmany(max(1, batch_size))
                    if not rows:
                        break
                    for r in rows:
                        yield {"segment_id": r[0], "model_name": r[1], "status": r[2], "payload": r[3]}

    def iter_artifacts(self, kind: str | None = None, batch_size: int = _FETCH_BATCH_SIZE) -> Iterator[dict[str, Any]]:
        sql = "select artifact_id, kind, path, status, payload_json from artifacts"
        params: tuple[Any, ...] = ()
        if kind is not None:
            sql += " where kind = %s"
            params = (kind,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                while True:
                    rows = cur.fetchmany(max(1, batch_size))
                    if not rows:
                        break
                    for r in rows:
                        yield {"artifact_id": r[0], "kind": r[1], "path": r[2], "status": r[3], "payload": r[4]}

    def _iter_payload_table(self, table: str, filter_column: str, filter_value: str | None, batch_size: int) -> Iterator[dict[str, Any]]:
        if table not in {"files", "segments"}:
            raise ValueError(f"Unsupported table: {table}")
        sql = f"select * from {table}"
        params: tuple[Any, ...] = ()
        if filter_value is not None:
            sql += f" where {filter_column} = %s"
            params = (filter_value,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                names = [d.name for d in cur.description]
                while True:
                    rows = cur.fetchmany(max(1, batch_size))
                    if not rows:
                        break
                    for row in rows:
                        data = dict(zip(names, row))
                        data["payload"] = data.get("payload_json", {})
                        yield data

    def _upsert_payload(self, table: str, pk_name: str, pk_value: str, columns: dict[str, Any], payload: dict[str, Any]) -> None:
        if table != "segments":
            raise ValueError("Unsupported Postgres upsert table")
        now = time.time()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into segments(segment_id, file_id, status, payload_json, updated_at)
                    values (%s, %s, %s, %s::jsonb, %s)
                    on conflict(segment_id) do update set
                        file_id=excluded.file_id,
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (pk_value, columns["file_id"], columns["status"], json.dumps(payload, sort_keys=True, default=str), now),
                )

    def _fetch_payload_table(self, table: str, status: str | None = None) -> list[dict[str, Any]]:
        if table not in {"files", "segments"}:
            raise ValueError(f"Unsupported table: {table}")
        sql = f"select * from {table}"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = %s"
            params = (status,)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                names = [d.name for d in cur.description]
                rows = cur.fetchall()
        out = []
        for row in rows:
            data = dict(zip(names, row))
            data["payload"] = data.get("payload_json", {})
            out.append(data)
        return out
