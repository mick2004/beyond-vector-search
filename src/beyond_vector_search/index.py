from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable

from .text import tokenize
from .types import Document


@dataclass(frozen=True)
class CorpusStats:
    vocab: set[str]
    df: dict[str, int]
    idf: dict[str, float]
    avg_dl: float
    doc_len: dict[str, int]
    rare_terms: set[str]


def build_corpus_stats(docs: list[Document], *, rare_df_threshold: int = 1) -> CorpusStats:
    df: dict[str, int] = {}
    doc_len: dict[str, int] = {}
    total_len = 0

    for d in docs:
        toks = tokenize(d.title + " " + d.text)
        doc_len[d.doc_id] = len(toks)
        total_len += len(toks)
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    n_docs = max(1, len(docs))
    avg_dl = total_len / n_docs

    # IDF with smoothing
    idf: dict[str, float] = {}
    for t, c in df.items():
        # BM25-style idf
        idf[t] = math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5))

    vocab = set(df.keys())
    rare_terms = {t for t, c in df.items() if c <= rare_df_threshold}

    return CorpusStats(vocab=vocab, df=df, idf=idf, avg_dl=avg_dl, doc_len=doc_len, rare_terms=rare_terms)


def term_freq(tokens: Iterable[str]) -> dict[str, int]:
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf



