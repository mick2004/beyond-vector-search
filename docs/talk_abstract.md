# Talk Abstract — Beyond Vector Search: Designing Agentic AI Systems with Learned Retrieval

Vector search solved “find similar text.” It did not solve “make correct decisions over a multi-step workflow.”

In agentic systems, retrieval is not a one-off step: it is a repeated input to planning, tool selection, and action. When retrieval fails early, the system doesn’t just answer incorrectly—it *compounds* error through subsequent steps, burning latency and cost while drifting further from the ground truth.

This talk reframes retrieval as a **decision problem**: for each query (and each step), the system should choose the retrieval strategy that best matches the query’s structure and the current failure modes of the stack. We’ll walk through an architecture for **learned/adaptive retrieval routing** using simple, production-friendly building blocks: multiple retrievers (vector + keyword), a router driven by query features and feedback, a lightweight evaluator, and telemetry to make the system debuggable.

Attendees will leave with a practical blueprint for what to build in the first 2–3 weeks: how to stand up the evaluation loop, how to log the right signals, how to introduce feedback safely, and how to measure progress without “benchmark theater.”

**Who it’s for**: senior data engineers, ML platform engineers, and architects building RAG or agentic AI systems in production.


