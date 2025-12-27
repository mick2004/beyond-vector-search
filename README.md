# beyond-vector-search

**Beyond vector search**: a tiny, runnable reference implementation of an **adaptive retrieval router** (vector vs keyword) with an **evaluation feedback loop** and **local SQLite telemetry**—built to illustrate how “learned retrieval” fits into agentic RAG systems.

- ![tests](https://github.com/mick2004/beyond-vector-search/actions/workflows/tests.yml/badge.svg)
- ![license](https://img.shields.io/badge/license-MIT-blue)

- Architecture diagram: open `diagrams/architecture.html` locally (or via GitHub Pages).
- Medium post draft: `docs/medium_post.md`
- No external downloads, no network calls, CPU-only.

## Quickstart (CPU, end-to-end)

Prereqs: **Python 3.11+**

### Option A (pip editable install)

```bash
python3 -m pip install -e .
python -m beyond_vector_search.run --query "What does BM25 do?"
python -m beyond_vector_search.evaluate
```

### Option B (uv)

```bash
uv venv
uv pip install -e .
python -m beyond_vector_search.run --query "What does BM25 do?"
python -m beyond_vector_search.evaluate
```

### Option C (Databricks notebook)

- Open `notebooks/databricks_demo.ipynb` in Databricks (Repos).
- It runs a single query, runs the offline evaluator (which updates router weights), and shows how to inspect the SQLite telemetry DB.
- By default on Databricks, the notebook configures **Delta telemetry tables** (Lakehouse) via env vars: `BVS_TELEMETRY=delta`.

## What this is / isn’t

- **This is**: a minimal system showing routing + retrievers + evaluation + feedback + logging.
- **This isn’t**: a production RAG stack, a benchmark, or an LLM-backed QA system (answering is template-based).

## What it does

- Two retrieval strategies:
  - **Vector**: TF‑IDF cosine similarity (dense-ish “vector retrieval” proxy, CPU-friendly).
  - **Keyword**: BM25-like scoring (term-frequency + IDF).
- **Router** chooses a strategy per query using:
  - query features (length, digit/ID presence, “rare token” ratio, etc.)
  - learned weights updated from past evaluation outcomes (simple bandit-style weight update)
- **Evaluation**: runs a labeled query set (bundled), logs results, and updates routing weights.
- **Logging**: SQLite DB at `runs/beyond_vector_search.sqlite`.

## Metrics (what to watch)

- **Retrieval accuracy proxy**: hit@k (did we retrieve the expected document?)
- **Answer exact match**: template answer vs expected (toy but deterministic)
- **Router quality**: win-rate of chosen strategy vs alternative; regret over time
- **Systems**: latency per stage (toy), routing distribution drift

## Repo layout

```
src/beyond_vector_search/   package code
data/                      toy corpus + labeled queries
diagrams/                  self-contained HTML architecture diagram
docs/                      talk abstract + FAQ + one-slide summary + checklists
tests/                     unit tests (stdlib unittest)
runs/                       created on first run (SQLite DB)
```

## Roadmap (pragmatic next steps)

- Add a **third** strategy (hybrid or metadata filter) and learn over more arms
- Replace toy evaluator with **LLM-graded** faithfulness / citation checks (offline)
- Introduce **per-tenant / per-domain** policies and drift detection
- Add **query clustering** and policy specialization (mixture-of-experts routing)

## License

MIT — see `LICENSE`.


