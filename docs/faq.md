# FAQ

## Is this a production RAG stack?

No. It’s a **systems reference**: multiple retrievers + routing + evaluation + feedback + telemetry, in the smallest runnable form.

## Why no external LLM call?

To keep the repo **offline and deterministic**. In production you’d replace the template answer generator with your LLM layer and keep the rest of the loop.

## Isn’t TF‑IDF “not vector search”?

In this repo, “vector” means **vector-space similarity**, implemented as **character n‑gram TF‑IDF cosine**. It’s intentionally dependency-free and “fuzzy” compared to token BM25-like scoring.

## How does the router “learn”?

The evaluator scores **both** strategies on labeled queries and updates **per-strategy weights** (a minimal bandit-style update).

## What’s the point of SQLite?

If you can’t **observe** router decisions and outcomes, you can’t debug or improve the system. SQLite keeps the feedback loop tangible.

## What would you change first for production?

- Replace the toy evaluator with real offline checks (citations/faithfulness, task success)
- Add guardrails: canary policies, weight bounds, drift alerts
- Add more strategies (hybrid, metadata filters, recency-biased retrievers)


