from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .answer import generate_answer
from .data import load_corpus, load_labels
from .evaluator import evaluate_run
from .index import build_corpus_stats
from .retrievers import HybridRetriever, KeywordRetriever, VectorRetriever
from .router import AdaptiveRouter
from .telemetry import telemetry_from_env


def evaluate_all(*, k: int = 5, db_path=None) -> dict:
    docs = load_corpus()
    labels = load_labels()

    stats = build_corpus_stats(docs, rare_df_threshold=1)
    vec = VectorRetriever.build(docs, stats)
    key = KeywordRetriever.build(docs, stats)
    hyb = HybridRetriever(docs=docs, vector=vec, keyword=key)

    store = telemetry_from_env(sqlite_path=db_path)
    router = AdaptiveRouter.build(vocab=stats.vocab, rare_terms=stats.rare_terms, db_path=db_path)

    per_query: list[dict] = []
    total = 0.0
    for lab in labels:
        top_vec = vec.search(lab.query, k=k)
        top_key = key.search(lab.query, k=k)
        top_hyb = hyb.search(lab.query, k=k)

        ans_vec = generate_answer(lab.query, top_vec).text
        ans_key = generate_answer(lab.query, top_key).text
        ans_hyb = generate_answer(lab.query, top_hyb).text

        s_vec = evaluate_run(
            top_k=top_vec,
            answer_text=ans_vec,
            expected_doc_id=lab.expected_doc_id,
            expected_answer=lab.expected_answer,
        )
        s_key = evaluate_run(
            top_k=top_key,
            answer_text=ans_key,
            expected_doc_id=lab.expected_doc_id,
            expected_answer=lab.expected_answer,
        )
        s_hyb = evaluate_run(
            top_k=top_hyb,
            answer_text=ans_hyb,
            expected_doc_id=lab.expected_doc_id,
            expected_answer=lab.expected_answer,
        )

        # What would the router choose right now?
        chosen, feats, route_meta = router.choose(lab.query)
        if chosen == "vector":
            chosen_scores = s_vec
        elif chosen == "keyword":
            chosen_scores = s_key
        else:
            chosen_scores = s_hyb

        total += chosen_scores.total

        router.update_from_scores(scores={"vector": s_vec.total, "keyword": s_key.total, "hybrid": s_hyb.total})

        store.log_run(
            query=lab.query,
            strategy=chosen,
            score=chosen_scores.total,
            meta={
                "eval": True,
                "query_id": lab.query_id,
                "expected_doc_id": lab.expected_doc_id,
                "features": asdict(feats),
                "route_meta": route_meta,
                "vector": {
                    "score_total": s_vec.total,
                    "hit_at_k": s_vec.hit_at_k,
                    "exact_match": s_vec.exact_match,
                    "top_doc_ids": [r.doc.doc_id for r in top_vec],
                },
                "keyword": {
                    "score_total": s_key.total,
                    "hit_at_k": s_key.hit_at_k,
                    "exact_match": s_key.exact_match,
                    "top_doc_ids": [r.doc.doc_id for r in top_key],
                },
                "hybrid": {
                    "score_total": s_hyb.total,
                    "hit_at_k": s_hyb.hit_at_k,
                    "exact_match": s_hyb.exact_match,
                    "top_doc_ids": [r.doc.doc_id for r in top_hyb],
                },
            },
        )

        per_query.append(
            {
                "query_id": lab.query_id,
                "query": lab.query,
                "chosen": chosen,
                "chosen_score": chosen_scores.total,
                "vector_score": s_vec.total,
                "keyword_score": s_key.total,
                "hybrid_score": s_hyb.total,
            }
        )

    mean = total / max(1, len(labels))
    state = router.load_state().to_json()
    return {"mean_score": mean, "n": len(labels), "router_state": state, "per_query": per_query}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate labeled queries and update router weights.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--db", type=str, default=None, help="Optional SQLite path")
    args = p.parse_args()
    out = evaluate_all(k=args.k, db_path=args.db)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


