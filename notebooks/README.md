# Notebooks

## Databricks Demo

**File:** `notebooks/databricks_lakebase_demo.py`

A fully documented, interactive notebook demonstrating the adaptive retrieval router.

### What it demonstrates

| Cell | What it does |
|------|--------------|
| 1 | Environment setup (repo root, Python path) |
| 2 | Telemetry backend configuration (Lakebase or SQLite) |
| 3 | Provision telemetry tables (`CREATE TABLE IF NOT EXISTS`) |
| 4 | Run a single query end-to-end (routing → retrieval → answer) |
| 5 | Offline evaluation loop (score all strategies, update router weights) |
| 6 | Inspect recent runs from telemetry |
| 7 | Inspect router state (learned weights) |

### Telemetry Backend Options

#### Option A: Databricks Lakebase (Postgres OLTP)

Best for production and persistent telemetry.

**Prerequisite:** Provision a Lakebase database instance first:
- [Lakebase provisioning docs](https://docs.databricks.com/aws/en/oltp/instances/create/)

Set environment variables:
```bash
BVS_TELEMETRY=lakebase
BVS_LAKEBASE_DSN=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

#### Option B: SQLite (local/dev)

No setup required. Uses a local SQLite file.

```bash
BVS_TELEMETRY=sqlite   # or just leave unset
```

The DB file is stored at `runs/beyond_vector_search.sqlite` by default.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ADAPTIVE RETRIEVAL ROUTER                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Query ──▶ Router (choose) ──▶ Retriever (keyword/vector/hybrid)       │
│                  │                           │                          │
│                  ▼                           ▼                          │
│          Feedback loop  ◀──────────  Evaluator (hit@k)                  │
│          (update weights)                                               │
│                  │                                                      │
│                  ▼                                                      │
│          Telemetry Store (Lakebase or SQLite)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Running Locally (without Databricks)

You can run the same code locally using the CLI:

```bash
# Quick run
python -m beyond_vector_search.run --query "INC-10010 cache stampede"

# Evaluation loop
python -m beyond_vector_search.evaluate
```

This uses SQLite by default. See `README.md` in repo root for full quickstart.
