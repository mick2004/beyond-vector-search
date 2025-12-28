# Databricks notebook source
# MAGIC %md
# MAGIC # Beyond Vector Search — Adaptive Retrieval Router Demo
# MAGIC
# MAGIC This notebook demonstrates an **adaptive retrieval router** that learns which retrieval strategy
# MAGIC works best for different query types.
# MAGIC
# MAGIC > **Note:** This notebook uses Databricks-native format (`# MAGIC`, `# COMMAND ----------`).
# MAGIC > It's designed to run in **Databricks Repos**. For non-Databricks environments, use the CLI instead:
# MAGIC > ```bash
# MAGIC > python -m beyond_vector_search.run --query "your query"
# MAGIC > python -m beyond_vector_search.evaluate
# MAGIC > ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                       ADAPTIVE RETRIEVAL ROUTER                             │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │   Query ──▶ Router ──▶ Retriever (keyword / vector / hybrid)                │
# MAGIC │               │                      │                                      │
# MAGIC │               ▼                      ▼                                      │
# MAGIC │         Feedback loop ◀───── Evaluator (hit@k)                              │
# MAGIC │         (update weights)                                                    │
# MAGIC │               │                                                             │
# MAGIC │               ▼                                                             │
# MAGIC │         Telemetry Store ◀── SQLite (default) or Lakebase Postgres           │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Telemetry Backend (choose one)
# MAGIC
# MAGIC | Backend | Setup | Best for |
# MAGIC |---------|-------|----------|
# MAGIC | **SQLite** (default) | No setup needed | Quick tests, local dev |
# MAGIC | **Lakebase Postgres** | [Provision instance first](https://docs.databricks.com/aws/en/oltp/instances/create/) | Production, shared state |
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Environment Setup
# MAGIC
# MAGIC This cell:
# MAGIC - Locates the repo root (so imports work without `pip install`)
# MAGIC - Adds `src/` to Python path
# MAGIC - Prints environment diagnostics

# COMMAND ----------

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from current directory until we find pyproject.toml (repo root marker)."""
    p = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root. Is pyproject.toml present?")


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"

# Make the package importable without pip install
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 60)
print("ENVIRONMENT SETUP")
print("=" * 60)
print(f"REPO_ROOT:      {REPO_ROOT}")
print(f"PYTHONPATH[0]:  {sys.path[0]}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Telemetry Backend Configuration
# MAGIC
# MAGIC Choose your telemetry backend:
# MAGIC
# MAGIC | Backend | When to use | Env vars |
# MAGIC |---------|-------------|----------|
# MAGIC | **Lakebase Postgres** | Production on Databricks | `BVS_TELEMETRY=lakebase`, `BVS_LAKEBASE_DSN=...` |
# MAGIC | **SQLite** | Local dev, quick tests | `BVS_TELEMETRY=sqlite` (default) |
# MAGIC
# MAGIC ### Lakebase Setup
# MAGIC
# MAGIC If you haven't provisioned Lakebase yet, follow:
# MAGIC https://docs.databricks.com/aws/en/oltp/instances/create/
# MAGIC
# MAGIC Then set your DSN below (prefer Databricks Secrets for credentials).

# COMMAND ----------

# ============================================================
# TELEMETRY BACKEND CONFIGURATION
# ============================================================
# Choose ONE of the options below by uncommenting the relevant section.

# --- OPTION A: Lakebase Postgres (Databricks OLTP) ---
# Requires a provisioned Lakebase instance. See:
# https://docs.databricks.com/aws/en/oltp/instances/create/

# os.environ["BVS_TELEMETRY"] = "lakebase"
# os.environ["BVS_LAKEBASE_DSN"] = "postgresql://USER:PASSWORD@HOST:5432/DBNAME"
# # Or use Databricks Secrets (recommended):
# # dsn = f"postgresql://{dbutils.secrets.get('scope','user')}:{dbutils.secrets.get('scope','pass')}@HOST:5432/DB"
# # os.environ["BVS_LAKEBASE_DSN"] = dsn

# --- OPTION B: SQLite (local / dev mode) ---
# No setup needed. Just uncomment:
os.environ["BVS_TELEMETRY"] = "sqlite"
# Optional: custom path (e.g., DBFS for persistence across cluster restarts)
# os.environ["BVS_DB_PATH"] = "/dbfs/tmp/beyond_vector_search.sqlite"

# --- Table names (applies to Lakebase only) ---
os.environ.setdefault("BVS_LAKEBASE_RUNS_TABLE", "beyond_vector_search_runs")
os.environ.setdefault("BVS_LAKEBASE_STATE_TABLE", "beyond_vector_search_router_state")

# Print current config
print("=" * 60)
print("TELEMETRY CONFIGURATION")
print("=" * 60)
backend = os.environ.get("BVS_TELEMETRY", "sqlite")
print(f"Backend:        {backend}")
if backend == "lakebase":
    print(f"DSN set:        {bool(os.environ.get('BVS_LAKEBASE_DSN'))}")
    print(f"Runs table:     {os.environ.get('BVS_LAKEBASE_RUNS_TABLE')}")
    print(f"State table:    {os.environ.get('BVS_LAKEBASE_STATE_TABLE')}")
else:
    db_path = os.environ.get("BVS_DB_PATH", str(REPO_ROOT / "runs" / "beyond_vector_search.sqlite"))
    print(f"SQLite path:    {db_path}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Provision Telemetry Tables
# MAGIC
# MAGIC This cell ensures the required tables exist:
# MAGIC - `beyond_vector_search_runs` — logs each retrieval run (query, strategy, score)
# MAGIC - `beyond_vector_search_router_state` — stores learned router weights
# MAGIC
# MAGIC For **Lakebase**: runs `CREATE TABLE IF NOT EXISTS` in Postgres.
# MAGIC For **SQLite**: creates the local DB file and tables.

# COMMAND ----------

from beyond_vector_search.telemetry import telemetry_from_env

# Initialize telemetry backend (Lakebase or SQLite based on env vars)
telemetry = telemetry_from_env()

# Trigger table creation by calling get_state (internally runs CREATE TABLE IF NOT EXISTS)
_ = telemetry.get_state("__init__", {})

backend = os.environ.get("BVS_TELEMETRY", "sqlite")
print("=" * 60)
print("TELEMETRY TABLES PROVISIONED")
print("=" * 60)
print(f"Backend: {backend}")
if backend == "lakebase":
    print(f"Tables created (if not exist):")
    print(f"  • {os.environ.get('BVS_LAKEBASE_RUNS_TABLE', 'beyond_vector_search_runs')}")
    print(f"  • {os.environ.get('BVS_LAKEBASE_STATE_TABLE', 'beyond_vector_search_router_state')}")
else:
    print("SQLite database initialized.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Run a Single Query (End-to-End)
# MAGIC
# MAGIC This cell demonstrates the full retrieval pipeline:
# MAGIC
# MAGIC 1. **Router** analyzes query features (digits, rare tokens, length)
# MAGIC 2. **Router** chooses a strategy: `keyword`, `vector`, or `hybrid`
# MAGIC 3. **Retriever** fetches top-k documents
# MAGIC 4. **Answerer** generates a template-based response
# MAGIC 5. **Telemetry** logs the run
# MAGIC
# MAGIC Try different queries to see how the router adapts:
# MAGIC - ID-heavy: `"INC-10010 cache stampede"` → likely **keyword**
# MAGIC - Natural language: `"how to fix slow queries"` → likely **vector**
# MAGIC - Mixed: `"explain error in TID-88410291"` → likely **hybrid**

# COMMAND ----------

from beyond_vector_search.run import run_once

# Example: ID-heavy query (router should prefer keyword or hybrid)
query = "pipeline failed for INC-10010 cache stampede"

result = run_once(query=query, k=5, db_path=None)

print("=" * 60)
print("SINGLE QUERY RESULT")
print("=" * 60)
print(f"Query:     {result['query']}")
print(f"Strategy:  {result['strategy']}")
print(f"Score:     {result['score']}")
print("-" * 60)
print("Top-K Documents:")
for i, doc in enumerate(result["top_k"], 1):
    print(f"  {i}. [{doc['doc_id']}] {doc['title']} (score: {doc['score']:.3f})")
print("-" * 60)
print("Answer:")
print(result["answer"])
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Offline Evaluation Loop (Learn from Labeled Data)
# MAGIC
# MAGIC This is the **feedback loop** that makes the router adaptive:
# MAGIC
# MAGIC 1. Load labeled queries (bundled in `data/labels.jsonl`)
# MAGIC 2. Run **all 3 strategies** on each query
# MAGIC 3. Score each strategy (hit@k — did we retrieve the expected doc?)
# MAGIC 4. **Update router weights**: reward the best-performing strategy
# MAGIC 5. Log everything to telemetry
# MAGIC
# MAGIC After running this cell multiple times, the router learns which strategy
# MAGIC works best for different query patterns.

# COMMAND ----------

from beyond_vector_search.evaluate import evaluate_all

report = evaluate_all(k=5, db_path=None)

print("=" * 60)
print("EVALUATION REPORT")
print("=" * 60)
print(f"Queries evaluated:  {report['n']}")
print(f"Mean score:         {report['mean_score']:.3f}")
print("-" * 60)
print("Router State (learned weights):")
for key, value in report["router_state"].items():
    print(f"  {key}: {value}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Inspect Telemetry — Recent Runs
# MAGIC
# MAGIC Query the telemetry store to see recent retrieval runs.
# MAGIC
# MAGIC Each row shows:
# MAGIC - `run_id`: auto-increment ID
# MAGIC - `ts_unix`: timestamp
# MAGIC - `strategy`: which retriever was used
# MAGIC - `score`: evaluation score (if labeled)
# MAGIC - `query`: the input query

# COMMAND ----------

import json

backend = os.environ.get("BVS_TELEMETRY", "sqlite")

if backend == "lakebase":
    # --- Lakebase Postgres ---
    dsn = os.environ["BVS_LAKEBASE_DSN"]
    runs_table = os.environ.get("BVS_LAKEBASE_RUNS_TABLE", "beyond_vector_search_runs")
    query = f"SELECT run_id, ts_unix, strategy, score, query FROM {runs_table} ORDER BY run_id DESC LIMIT 10"

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except Exception:
        import psycopg2
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

    print("=" * 60)
    print("RECENT RUNS (Lakebase Postgres)")
    print("=" * 60)
    for row in rows:
        print(f"  {row}")

else:
    # --- SQLite ---
    import sqlite3
    db_path = os.environ.get("BVS_DB_PATH", str(REPO_ROOT / "runs" / "beyond_vector_search.sqlite"))

    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT run_id, ts_unix, strategy, score, query FROM runs ORDER BY run_id DESC LIMIT 10")
        rows = cur.fetchall()
        conn.close()

        print("=" * 60)
        print("RECENT RUNS (SQLite)")
        print("=" * 60)
        for row in rows:
            print(f"  {row}")
    else:
        print("No runs yet. Run cells 4 or 5 first.")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Inspect Telemetry — Router State
# MAGIC
# MAGIC The router stores its learned weights in the `router_state` table.
# MAGIC
# MAGIC Key fields:
# MAGIC - `weight_keyword`: learned preference for keyword retrieval
# MAGIC - `weight_vector`: learned preference for vector retrieval
# MAGIC - `weight_hybrid`: learned preference for hybrid retrieval
# MAGIC - `lr`: learning rate for weight updates

# COMMAND ----------

backend = os.environ.get("BVS_TELEMETRY", "sqlite")

if backend == "lakebase":
    # --- Lakebase Postgres ---
    dsn = os.environ["BVS_LAKEBASE_DSN"]
    state_table = os.environ.get("BVS_LAKEBASE_STATE_TABLE", "beyond_vector_search_router_state")
    query = f"SELECT key, value_json FROM {state_table} ORDER BY key"

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except Exception:
        import psycopg2
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

    print("=" * 60)
    print("ROUTER STATE (Lakebase Postgres)")
    print("=" * 60)
    for key, val in rows:
        print(f"  {key}:")
        parsed = json.loads(val) if isinstance(val, str) else val
        for k, v in parsed.items():
            print(f"    {k}: {v}")

else:
    # --- SQLite ---
    import sqlite3
    db_path = os.environ.get("BVS_DB_PATH", str(REPO_ROOT / "runs" / "beyond_vector_search.sqlite"))

    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT key, value_json FROM router_state ORDER BY key")
        rows = cur.fetchall()
        conn.close()

        print("=" * 60)
        print("ROUTER STATE (SQLite)")
        print("=" * 60)
        for key, val in rows:
            print(f"  {key}:")
            parsed = json.loads(val)
            for k, v in parsed.items():
                print(f"    {k}: {v}")
    else:
        print("No state yet. Run cell 5 (evaluation) first.")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Try different queries** in Cell 4 to see routing in action
# MAGIC 2. **Run Cell 5 multiple times** to watch the router learn
# MAGIC 3. **Inspect telemetry** (Cells 6-7) to see how weights evolve
# MAGIC
# MAGIC ### To reset learning
# MAGIC - **Lakebase**: `TRUNCATE TABLE beyond_vector_search_runs; TRUNCATE TABLE beyond_vector_search_router_state;`
# MAGIC - **SQLite**: Delete `runs/beyond_vector_search.sqlite`
# MAGIC
# MAGIC ### Resources
# MAGIC - [Lakebase provisioning docs](https://docs.databricks.com/aws/en/oltp/instances/create/)
# MAGIC - [GitHub repo](https://github.com/mick2004/beyond-vector-search)
