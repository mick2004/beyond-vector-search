from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .answer import build_context, generate_answer
from .data import load_corpus, load_labels
from .evaluator import evaluate_run
from .index import build_corpus_stats
from .retrievers import HybridRetriever, KeywordRetriever, VectorRetriever
from .router import AdaptiveRouter
from .telemetry import telemetry_from_env


def run_once(*, query: str, k: int = 5, db_path=None) -> dict:
    docs = load_corpus()
    labels = {l.query: l for l in load_labels()}

    stats = build_corpus_stats(docs, rare_df_threshold=1)
    vec = VectorRetriever.build(docs, stats)
    key = KeywordRetriever.build(docs, stats)
    hyb = HybridRetriever(docs=docs, vector=vec, keyword=key)

    router = AdaptiveRouter.build(vocab=stats.vocab, rare_terms=stats.rare_terms, db_path=db_path)
    strategy, feats, route_meta = router.choose(query)

    if strategy == "vector":
        top_k = vec.search(query, k=k)
    elif strategy == "keyword":
        top_k = key.search(query, k=k)
    else:
        top_k = hyb.search(query, k=k)
    ctx = build_context(top_k)
    ans = generate_answer(query, top_k)

    score = 0.0
    expected = None
    if query in labels:
        expected = labels[query]
        scores = evaluate_run(
            top_k=top_k,
            answer_text=ans.text,
            expected_doc_id=expected.expected_doc_id,
            expected_answer=expected.expected_answer,
        )
        score = scores.total

    store = telemetry_from_env(sqlite_path=db_path)
    store.log_run(
        query=query,
        strategy=strategy,
        score=score,
        meta={
            "k": k,
            "features": asdict(feats),
            "route_meta": route_meta,
            "top_doc_ids": [r.doc.doc_id for r in top_k],
            "context_preview": ctx[:240],
        },
    )

    out = {
        "query": query,
        "strategy": strategy,
        "top_k": [{"doc_id": r.doc.doc_id, "title": r.doc.title, "score": r.score} for r in top_k],
        "answer": ans.text,
        "score": score,
        "labeled": query in labels,
    }
    if expected is not None:
        out["expected_doc_id"] = expected.expected_doc_id
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run the adaptive retrieval router (toy, CPU-only).")
    p.add_argument("--query", required=True, type=str, help="User query")
    p.add_argument("--k", type=int, default=5, help="Top-k passages")
    p.add_argument("--db", type=str, default=None, help="Optional SQLite path")
    args = p.parse_args()

    out = run_once(query=args.query, k=args.k, db_path=args.db)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


