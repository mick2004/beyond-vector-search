from __future__ import annotations

from dataclasses import dataclass

from .types import RetrievalResult


@dataclass(frozen=True)
class EvalScores:
    hit_at_k: float
    exact_match: float

    @property
    def total(self) -> float:
        # keep it simple and deterministic; weights sum to 1
        return 0.7 * self.hit_at_k + 0.3 * self.exact_match


def score_retrieval(top_k: list[RetrievalResult], expected_doc_id: str) -> float:
    return 1.0 if any(r.doc.doc_id == expected_doc_id for r in top_k) else 0.0


def score_answer(answer: str, expected_answer: str) -> float:
    # Toy exact match (case-insensitive, whitespace-normalized)
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    return 1.0 if norm(answer) == norm(expected_answer) else 0.0


def evaluate_run(
    *,
    top_k: list[RetrievalResult],
    answer_text: str,
    expected_doc_id: str,
    expected_answer: str,
) -> EvalScores:
    hit = score_retrieval(top_k, expected_doc_id)
    em = score_answer(answer_text, expected_answer)
    return EvalScores(hit_at_k=hit, exact_match=em)


