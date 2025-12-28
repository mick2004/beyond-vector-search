# Notebooks

## `demo.py` — Databricks Notebook

**File:** `notebooks/demo.py`

A fully documented, interactive notebook demonstrating the adaptive retrieval router.

> **⚠️ Databricks-specific:** This notebook uses Databricks-native format (`# MAGIC`, `# COMMAND ----------`).
> It's designed to run in **Databricks Repos**. For non-Databricks environments, use the CLI instead (see below).

### What it demonstrates

| Cell | What it does |
|------|--------------|
| 1 | Environment setup (repo root, Python path) |
| 2 | Telemetry backend configuration (SQLite or Lakebase) |
| 3 | Provision telemetry tables (`CREATE TABLE IF NOT EXISTS`) |
| 4 | Run a single query end-to-end (routing → retrieval → answer) |
| 5 | Offline evaluation loop (score all strategies, update router weights) |
| 6 | Inspect recent runs from telemetry |
| 7 | Inspect router state (learned weights) |

### Telemetry Backend Options

| Backend | Setup | Best for |
|---------|-------|----------|
| **SQLite** (default) | No setup needed | Quick tests, local dev |
| **Lakebase Postgres** | [Provision instance first](https://docs.databricks.com/aws/en/oltp/instances/create/) | Production, shared state |

---

## Running Without Databricks (CLI)

If you're **not using Databricks**, you can run the same functionality via the command line on any machine with Python 3.11+:

```bash
# Install
pip install -e .

# Run a single query
python -m beyond_vector_search.run --query "INC-10010 cache stampede"

# Run offline evaluation (updates router weights)
python -m beyond_vector_search.evaluate
```

This uses SQLite by default. Telemetry is stored in `runs/beyond_vector_search.sqlite`.

See the main `README.md` for full quickstart instructions.
