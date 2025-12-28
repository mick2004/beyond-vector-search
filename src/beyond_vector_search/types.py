from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


Strategy = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    doc: Document
    score: float


@dataclass(frozen=True)
class QueryLabel:
    query_id: str
    query: str
    expected_doc_id: str
    expected_answer: str


@dataclass(frozen=True)
class RunOutput:
    query: str
    strategy: Strategy
    top_k: Sequence[RetrievalResult]
    answer: str
    score: float
    expected_doc_id: str | None = None


