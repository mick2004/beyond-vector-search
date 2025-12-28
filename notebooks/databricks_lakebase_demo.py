# Databricks notebook source
# beyond-vector-search — Databricks demo (Lakebase Postgres telemetry)
#
# IMPORTANT: Databricks Repos treats notebooks with the same base name as conflicting.
# This notebook is intentionally named `databricks_lakebase_demo.py` (not `databricks_demo.*`).
#
# It demonstrates:
# - Adaptive retrieval routing (keyword / vector / hybrid)
# - Offline evaluation loop that updates router weights
# - Telemetry persisted to Lakebase Postgres (OLTP)

# COMMAND ----------
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking up until pyproject.toml is found."""
    p = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root (pyproject.toml not found).")


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("REPO_ROOT:", REPO_ROOT)
print("PYTHONPATH[0]:", sys.path[0])

# COMMAND ----------
# --- Lakebase Postgres telemetry configuration ---
#
# Set these as cluster environment variables (recommended), or in this cell:
# - BVS_TELEMETRY=lakebase
# - BVS_LAKEBASE_DSN=postgresql://USER:PASSWORD@HOST:5432/DBNAME
#
# Tip: store USER/PASSWORD in Databricks Secrets and build the DSN here.

os.environ.setdefault("BVS_TELEMETRY", "lakebase")

# TODO: set your DSN. Prefer secrets:
# dsn = f"postgresql://{dbutils.secrets.get('scope','user')}:{dbutils.secrets.get('scope','pass')}@HOST:5432/DB"
# os.environ["BVS_LAKEBASE_DSN"] = dsn

os.environ.setdefault("BVS_LAKEBASE_RUNS_TABLE", "beyond_vector_search_runs")
os.environ.setdefault("BVS_LAKEBASE_STATE_TABLE", "beyond_vector_search_router_state")

print("BVS_TELEMETRY:", os.environ.get("BVS_TELEMETRY"))
print("DSN set:", bool(os.environ.get("BVS_LAKEBASE_DSN")))
print("Runs table:", os.environ.get("BVS_LAKEBASE_RUNS_TABLE"))
print("State table:", os.environ.get("BVS_LAKEBASE_STATE_TABLE"))

# COMMAND ----------
from beyond_vector_search.run import run_once

out = run_once(query="pipeline failed for INC-10010 cache stampede", k=5, db_path=None)
out

# COMMAND ----------
from beyond_vector_search.evaluate import evaluate_all

report = evaluate_all(k=5, db_path=None)
{
    "mean_score": report["mean_score"],
    "n": report["n"],
    "router_state": report["router_state"],
}

# COMMAND ----------
# Inspect the most recent runs (Lakebase Postgres)
dsn = os.environ["BVS_LAKEBASE_DSN"]
runs_table = os.environ.get("BVS_LAKEBASE_RUNS_TABLE", "beyond_vector_search_runs")

query = f"SELECT run_id, ts_unix, strategy, score, query FROM {runs_table} ORDER BY run_id DESC LIMIT 10"

try:
    import psycopg  # type: ignore

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    rows
except Exception:
    import psycopg2  # type: ignore

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    rows

# COMMAND ----------
# Inspect the current router state (Lakebase Postgres)
state_table = os.environ.get("BVS_LAKEBASE_STATE_TABLE", "beyond_vector_search_router_state")

query = f"SELECT key, value_json FROM {state_table} ORDER BY key"

try:
    import psycopg  # type: ignore

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    rows
except Exception:
    import psycopg2  # type: ignore

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    rows


