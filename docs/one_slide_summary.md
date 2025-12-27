# One-slide summary (copy/paste)

## Beyond Vector Search: Learned Retrieval for Agentic AI

**Problem**: In agentic workflows, retrieval errors compound across steps → wrong plans, wrong tools, higher cost.

**Thesis**: Retrieval is a *decision layer*, not a single algorithm.

**Architecture**:
- Query → **Router** (features + learned bias)
- Router → **Vector retriever** *or* **Keyword/BM25 retriever**
- Top‑k → Context builder → Answer generator
- Offline: Evaluator → Feedback store → Router (policy update)
- Always: Telemetry (query, strategy, score)

**Metrics I’d measure**:
- hit@k for “did we fetch the right evidence?”
- answer exact match / faithfulness proxy
- router regret: chosen arm vs best arm (offline)
- latency + cost per step (agent loops)

**Result**: The system learns which retrieval strategy to trust *for this kind of query* and *this current corpus*.


