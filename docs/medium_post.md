# Beyond Vector Search: Designing Agentic AI Systems with Learned Retrieval

**Subtitle:** Retrieval isn’t a component anymore—it’s a decision layer with feedback, telemetry, and guardrails.

---

## 1) The uncomfortable truth: static retrieval breaks in agentic workflows

Vector search + vanilla RAG works surprisingly well when the workflow is: “one query, one retrieval, one answer.”

Agentic systems aren’t that workflow.

Agents retrieve **multiple times** across a plan: to choose tools, to fill parameters, to validate intermediate state, to recover from errors. A small miss early doesn’t just produce a slightly worse answer—it becomes a **compounding error**:

- **Bad context → wrong plan**: the agent selects the wrong tool or wrong sequence.
- **Wrong plan → worse queries**: the next retrieval is conditioned on earlier mistakes.
- **Worse queries → wider drift**: the system spends latency and cost walking away from ground truth.

In this regime, “better embeddings” is not a complete strategy. You need a system that can decide **how** to retrieve for each step, and improve that decision over time.

---

## 2) Learned/adaptive retrieval, defined as a systems decision problem

In systems terms, *learned retrieval* is not “a fancy retriever.” It’s the framing that retrieval is a **policy**:

\[
\pi(\text{query}, \text{context}, \text{history}) \rightarrow \text{retrieval strategy}
\]

Where “strategy” could be:

- keyword/BM25-like search
- vector similarity search
- hybrid retrieval
- metadata filters
- recency-biased search
- graph traversal / entity retrieval

The core move is to make that choice **explicit**, and to back it with:

- **features you can compute at runtime**
- **telemetry you can audit**
- **evaluation feedback that updates the policy**

That’s what this repo demonstrates in minimal form: **router → (vector | keyword) → context → answer**, with a small evaluator that updates routing weights and logs everything to SQLite.

---

## 3) Reference architecture: retrieval routing + feedback loop

The architecture is intentionally small, but it mirrors the shape of real systems.

- **Diagram 1 (overview)**: Open `diagrams/architecture.html` (self-contained HTML).
  - Flow: *User Query → Router → (Vector Retriever / Keyword Retriever) → Context Builder → Answer Generator*
  - Loop: *Evaluator → Feedback Store → Router* (policy update)
  - Sidecar: *Logging/Telemetry store*

In production, the “Answer Generator” would be your LLM layer. The important point is that the router and feedback loop sit *around* it, turning retrieval into a measurable decision layer instead of a fixed dependency.

---

## Selected code paths (minimal but real)

This project is intentionally small, so you can point to the exact lines where “learned retrieval” lives.

**Router decision (features + learned bias → strategy):**

```python
# src/beyond_vector_search/router.py
strategy, feats, meta = router.choose(query)

heuristic_keyword = (
    1.25 * feats.digit_ratio
    + 1.00 * feats.oov_ratio
    + 1.25 * feats.rare_ratio
    + (0.10 if feats.n_tokens <= 3 else 0.0)
)
heuristic_vector = 0.50 * (1.0 - min(1.0, feats.oov_ratio + feats.rare_ratio))

score_keyword = heuristic_keyword + state.weight_keyword
score_vector = heuristic_vector + state.weight_vector
strategy = "keyword" if score_keyword >= score_vector else "vector"
```

**Feedback update (bandit-style push toward the better arm):**

```python
# src/beyond_vector_search/router.py
if score_keyword > score_vector:
    st.weight_keyword += st.lr
    st.weight_vector -= st.lr
elif score_vector > score_keyword:
    st.weight_vector += st.lr
    st.weight_keyword -= st.lr
```

**Offline evaluation loop (score both, then update + log):**

```python
# src/beyond_vector_search/evaluate.py
top_vec = vec.search(lab.query, k=k)
top_key = key.search(lab.query, k=k)

s_vec = evaluate_run(...)
s_key = evaluate_run(...)

chosen, feats, route_meta = router.choose(lab.query)
router.update_from_pairwise(score_vector=s_vec.total, score_keyword=s_key.total)
db.log_run(query=lab.query, strategy=chosen, score=chosen_scores.total, meta={...})
```

---

## 4) Why two retrievers are the minimum viable reality check

If you only have one retrieval strategy, you can’t learn routing—you can only learn “how to tune that one thing.”

Two strategies create disagreement, and disagreement is where learning becomes visible:

- **Keyword/BM25-like** tends to dominate when:
  - the query includes IDs, codes, incident numbers, tickets
  - the important signal is in rare tokens (“exactness matters”)
- **Vector-ish similarity** tends to dominate when:
  - users paraphrase, abbreviate, or change formatting
  - the important signal is semantic and distributed

This repo uses:

- **Keyword retriever**: BM25-like scoring (token exact-match bias)
- **Vector retriever**: character n-gram TF‑IDF cosine similarity (cheap “fuzzy” vector proxy)

That pairing is deliberate: it creates realistic trade-offs without external services or heavy dependencies.

---

## 5) Failure modes and trade-offs (what breaks when you add learning)

Adding a router + feedback loop makes retrieval *more powerful*, but it also creates new failure modes. A few that matter in real deployments:

- **Latency overhead**
  - Scoring multiple retrievers increases compute and tail latency.
  - Mitigation: route first using cheap features, run only the chosen retriever online; score both arms offline.

- **Evaluation noise**
  - If the evaluator is weak or inconsistent, it will push the policy in the wrong direction.
  - Mitigation: start with deterministic proxies + curated labeled sets; gate changes; measure regret.

- **Policy oscillation**
  - Small datasets or shifting traffic patterns can cause “thrash.”
  - Mitigation: bounded updates, smoothing, minimum data thresholds, canarying.

- **Mismatch between offline and online reality**
  - Your offline queries may not represent agent steps, tool calls, or user behavior.
  - Mitigation: log real traces, sample them, label the hard cases, and continuously refresh eval sets.

- **Metric gaming**
  - If the policy optimizes an easy proxy (e.g., exact match), it may hurt real task success.
  - Mitigation: layer metrics: evidence retrieval quality + answer faithfulness + task success.

---

## 6) Implementation Strategy (2–3 weeks): build order that ships

If you’re starting from a working RAG/agent system, the first iteration should be about **observability and controllability**, not “learning” in the ML sense.

Here’s a realistic build order:

- **Week 1: make retrieval decisions visible**
  - Introduce a router interface (even if it’s heuristic-only).
  - Log: query, chosen strategy, top‑k doc IDs, latency, and a “success” proxy.

- **Week 2: introduce offline evaluation**
  - Build a labeled set from real traces (plus synthetic ID-heavy cases).
  - Score both strategies offline.
  - Track: win-rate of each strategy by query segment (IDs vs natural language, short vs long).

- **Week 3: close the feedback loop carefully**
  - Add policy updates (bandit-style or weighted scoring) behind guardrails.
  - Roll out as: offline-only → shadow mode → canary → default.

This repo is a toy, but it includes the end-to-end mechanics: an evaluator that scores both retrievers and updates router weights, plus SQLite telemetry so you can inspect outcomes.

---

## 7) Measurement + Guardrails (what to track so it doesn’t drift)

If you want learned retrieval to improve an agent, measure at multiple layers:

- **Evidence quality**
  - hit@k / recall@k on labeled evidence
  - citation coverage (does the final answer cite retrieved evidence?)

- **Answer quality**
  - faithfulness / groundedness checks
  - exact match (only for toy or strict domains)

- **Router quality**
  - regret: chosen strategy vs best strategy (offline)
  - calibration: confidence vs outcomes
  - stability: how often policy flips week-over-week

- **System health**
  - p95/p99 latency per stage
  - cost per successful task (especially in multi-step loops)
  - drift signals: routing distribution changes by segment

---

## 8) Proof (so it doesn’t stay conceptual)

- **Repo**: `beyond-vector-search` (toy but runnable end-to-end, CPU-only)
- **Architecture diagram included**: `diagrams/architecture.html`
- **Next**: extend evaluator + learned router using your real traces and task-success metrics

---

## Conclusion: treat retrieval like a policy, not a dependency

Agentic AI systems don’t fail because vector search is “bad.” They fail because the system treats retrieval as a static primitive in a dynamic, multi-step decision process.

If you want agents that improve over time, you need:

- multiple retrieval strategies
- a router that can explain its choice
- telemetry that makes outcomes inspectable
- an evaluation loop that updates policy safely

If you want a minimal reference implementation to start from, run the repo and open the diagram: `diagrams/architecture.html`.

---

## Further Reading (topics/papers)

- Multi-armed bandits for decision-making under uncertainty
- Learning to retrieve / query routing
- Hybrid retrieval (BM25 + dense) and late interaction models
- RAG evaluation: faithfulness/groundedness, citation-based checks
- Agent evaluation: task success metrics, trajectory scoring, tool-use reliability


