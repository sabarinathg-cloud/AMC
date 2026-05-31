from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any


class SQLiteStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self.connect()) as conn:
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
        with closing(self.connect()) as conn:
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
        with closing(self.connect()) as conn:
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

    def upsert_model_result(self, segment_id: str, model_name: str, status: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with closing(self.connect()) as conn:
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

    def record_artifact(self, artifact_id: str, kind: str, path: Path, status: str, payload: dict[str, Any] | None = None) -> None:
        now = time.time()
        with closing(self.connect()) as conn:
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

    def record_failure(self, failure_id: str, scope: str, scope_id: str, error: str, retryable: bool = True, traceback: str | None = None) -> None:
        with closing(self.connect()) as conn:
            conn.execute(
                """
                insert or replace into failures(failure_id, scope, scope_id, error, traceback, retryable, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (failure_id, scope, scope_id, error, traceback, int(retryable), time.time()),
            )

    def request_pause(self, run_id: str, worker_id: str | None = None, global_pause: bool = False) -> None:
        with closing(self.connect()) as conn:
            conn.execute(
                "insert into pause_requests(run_id, worker_id, global_pause, created_at) values (?, ?, ?, ?)",
                (run_id, worker_id, int(global_pause), time.time()),
            )

    def should_pause(self, run_id: str, worker_id: str | None = None) -> bool:
        with closing(self.connect()) as conn:
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
        with closing(self.connect()) as conn:
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
        with closing(self.connect()) as conn:
            return int(conn.execute(f"select count(*) from {table}").fetchone()[0])

    def fetch_files(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from files"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = ?"
            params = (status,)
        with closing(self.connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_segments(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from segments"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " where status = ?"
            params = (status,)
        with closing(self.connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_model_results(self, segment_id: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from model_results"
        params: tuple[Any, ...] = ()
        if segment_id is not None:
            sql += " where segment_id = ?"
            params = (segment_id,)
        with closing(self.connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def fetch_artifacts(self, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "select * from artifacts"
        params: tuple[Any, ...] = ()
        if kind is not None:
            sql += " where kind = ?"
            params = (kind,)
        with closing(self.connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

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
