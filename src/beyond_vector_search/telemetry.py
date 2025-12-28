from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .data import repo_root
from .types import Strategy


def default_sqlite_path() -> Path:
    return repo_root() / "runs" / "beyond_vector_search.sqlite"


class TelemetryStore:
    """
    Minimal persistence interface used by the router + runner.

    Implementations:
    - SQLite (default, local/offline)
    - Databricks Lakebase Postgres (OLTP)
    """

    def log_run(self, *, query: str, strategy: Strategy, score: float, meta: dict, ts_unix: float | None = None) -> None:
        raise NotImplementedError

    def get_state(self, key: str, default: dict) -> dict:
        raise NotImplementedError

    def set_state(self, key: str, value: dict) -> None:
        raise NotImplementedError


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_unix REAL NOT NULL,
  query TEXT NOT NULL,
  strategy TEXT NOT NULL,
  score REAL NOT NULL,
  meta_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS router_state (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL
);
"""


@dataclass
class SQLiteTelemetry(TelemetryStore):
    path: str | Path

    def _path_str(self) -> str:
        return str(self.path)

    def _connect(self) -> sqlite3.Connection:
        p = self._path_str()
        if p != ":memory:":
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SQLITE_SCHEMA)
        return conn

    def log_run(self, *, query: str, strategy: Strategy, score: float, meta: dict, ts_unix: float | None = None) -> None:
        ts_unix = ts_unix if ts_unix is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs(ts_unix, query, strategy, score, meta_json) VALUES(?,?,?,?,?)",
                (ts_unix, query, strategy, float(score), json.dumps(meta, sort_keys=True)),
            )

    def get_state(self, key: str, default: dict) -> dict:
        with self._connect() as conn:
            cur = conn.execute("SELECT value_json FROM router_state WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return default
            return json.loads(row[0])

    def set_state(self, key: str, value: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO router_state(key, value_json) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, json.dumps(value, sort_keys=True)),
            )


@dataclass
class LakebasePostgresTelemetry(TelemetryStore):
    """
    Databricks Lakebase Postgres backend (OLTP).

    Requires a PostgreSQL driver in the environment (psycopg or psycopg2).
    Imports happen lazily so local runs remain dependency-free.
    """

    dsn: str
    runs_table: str = "beyond_vector_search_runs"
    state_table: str = "beyond_vector_search_router_state"

    def _connect(self):
        # psycopg (v3) preferred; fall back to psycopg2.
        try:
            import psycopg  # type: ignore

            return psycopg.connect(self.dsn)
        except Exception:
            try:
                import psycopg2  # type: ignore

                return psycopg2.connect(self.dsn)
            except Exception as e:
                raise RuntimeError(
                    "LakebasePostgresTelemetry requires a PostgreSQL driver. "
                    "Install psycopg (v3) or psycopg2 in your Databricks cluster."
                ) from e

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.runs_table} (
                      run_id BIGSERIAL PRIMARY KEY,
                      ts_unix DOUBLE PRECISION NOT NULL,
                      query TEXT NOT NULL,
                      strategy TEXT NOT NULL,
                      score DOUBLE PRECISION NOT NULL,
                      meta_json JSONB NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.state_table} (
                      key TEXT PRIMARY KEY,
                      value_json JSONB NOT NULL
                    );
                    """
                )
                conn.commit()

    def log_run(self, *, query: str, strategy: Strategy, score: float, meta: dict, ts_unix: float | None = None) -> None:
        ts_unix = ts_unix if ts_unix is not None else time.time()
        self._ensure_tables()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.runs_table}(ts_unix, query, strategy, score, meta_json) VALUES (%s,%s,%s,%s,%s)",
                    (float(ts_unix), query, strategy, float(score), json.dumps(meta, sort_keys=True)),
                )
                conn.commit()

    def get_state(self, key: str, default: dict) -> dict:
        self._ensure_tables()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT value_json FROM {self.state_table} WHERE key = %s", (key,))
                row = cur.fetchone()
                if not row:
                    return default
                val = row[0]
                if isinstance(val, str):
                    return json.loads(val)
                return val

    def set_state(self, key: str, value: dict) -> None:
        self._ensure_tables()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.state_table}(key, value_json)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value_json = EXCLUDED.value_json
                    """,
                    (key, json.dumps(value, sort_keys=True)),
                )
                conn.commit()


def telemetry_from_env(*, sqlite_path: str | Path | None = None) -> TelemetryStore:
    """
    Select telemetry backend using env vars.

    - Default: SQLite at `sqlite_path` (or repo runs/ path)
    - Databricks Lakebase Postgres: set BVS_TELEMETRY=lakebase and provide:
        - BVS_LAKEBASE_DSN (Postgres connection string)
        - Optional: BVS_LAKEBASE_RUNS_TABLE, BVS_LAKEBASE_STATE_TABLE
    """
    backend = os.environ.get("BVS_TELEMETRY", "sqlite").strip().lower()
    if backend == "lakebase":
        dsn = os.environ.get("BVS_LAKEBASE_DSN")
        if not dsn:
            raise RuntimeError(
                "BVS_TELEMETRY=lakebase requires BVS_LAKEBASE_DSN "
                "(a PostgreSQL connection string for your Lakebase Postgres database)."
            )
        runs_table = os.environ.get("BVS_LAKEBASE_RUNS_TABLE", "beyond_vector_search_runs")
        state_table = os.environ.get("BVS_LAKEBASE_STATE_TABLE", "beyond_vector_search_router_state")
        return LakebasePostgresTelemetry(dsn=dsn, runs_table=runs_table, state_table=state_table)
    return SQLiteTelemetry(path=sqlite_path or default_sqlite_path())


