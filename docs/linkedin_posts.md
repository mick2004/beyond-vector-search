# LinkedIn drafts

## Short (hooky)

Static vector search is fine… until you build an agent.

Agents don’t “retrieve once.” They retrieve, plan, act, retrieve again. One bad retrieval early can compound across steps and burn cost fast.

I built a tiny reference repo showing **learned/adaptive retrieval routing**: choose keyword vs vector per query, log outcomes, and update the router via an evaluation loop.

Repo + diagram included.

## Medium (technical)

Most RAG stacks treat retrieval as a fixed component: embed → similarity search → top‑k → prompt.

Agentic systems break that assumption. Retrieval becomes a repeated decision under uncertainty, and the “right” retriever depends on the query:
- IDs / rare tokens → keyword/BM25 often wins
- fuzzy variants / formatting differences → vector-ish similarity can win

So the architecture needs a **router**:
- featurize query (length, digit/ID signals, OOV/rarity)
- choose among strategies
- log strategy + outcome
- run offline eval to update policy weights (bandit-style)

I put a minimal runnable version in `beyond-vector-search` with a self-contained HTML diagram.

## Long (story + lessons)

I kept seeing the same failure pattern in agentic systems:

Step 1: retrieve slightly-wrong context  
Step 2: plan based on that context  
Step 3: call tools with the wrong parameters  
Step 4: retrieve again… but now the query is biased by earlier mistakes  

It’s not “one hallucination.” It’s **compounding error**, and it gets worse as you add tools and steps.

The fix isn’t “better embeddings.” The fix is a systems move: treat retrieval as a **decision layer**.

What that means in practice:
- run multiple retrieval strategies (keyword + vector, at minimum)
- add a router that picks per query based on signals you can observe
- create an evaluation loop that scores both strategies and updates the router
- log everything (query, strategy, top‑k IDs, score) so you can debug and iterate

I built a small, offline, CPU-only reference repo implementing exactly this loop (toy but real). It includes an HTML architecture diagram you can drop into a blog post or CFP.

If you’re building agentic RAG, I’d love feedback on what you’d measure and how you’d gate policy updates.


