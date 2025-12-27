from __future__ import annotations

from dataclasses import dataclass

from .text import join_top_sentences
from .types import RetrievalResult


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]


def build_context(top_k: list[RetrievalResult], *, max_chars: int = 900) -> str:
    chunks: list[str] = []
    used = 0
    for r in top_k:
        snippet = join_top_sentences(r.doc.text, max_sentences=2)
        block = f"[{r.doc.doc_id}] {r.doc.title}: {snippet}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def generate_answer(query: str, top_k: list[RetrievalResult]) -> Answer:
    if not top_k:
        return Answer(text="I couldn't find relevant context in the toy corpus.", citations=[])
    top = top_k[0].doc
    snippet = join_top_sentences(top.text, max_sentences=2)
    txt = (
        f"Based on the retrieved context, here's the best match:\n\n"
        f"{top.title}\n"
        f"{snippet}\n\n"
        f"(Query: {query})"
    )
    return Answer(text=txt, citations=[top.doc_id])


