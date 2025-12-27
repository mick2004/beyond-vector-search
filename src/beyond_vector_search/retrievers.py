from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .index import CorpusStats, term_freq
from .text import stable_topk, tokenize
from .types import Document, RetrievalResult


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _l2norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def _tfidf_vector(tokens: list[str], stats: CorpusStats) -> dict[str, float]:
    tf = term_freq(tokens)
    vec: dict[str, float] = {}
    for t, c in tf.items():
        if t not in stats.idf:
            continue
        # sublinear tf; idf from corpus stats
        vec[t] = (1.0 + math.log(c)) * stats.idf[t]
    return vec


_WS_RE = re.compile(r"\s+")


def _char_ngrams(text: str, *, n: int = 4) -> list[str]:
    """
    Character n-grams provide a cheap "fuzzy" signal (typos, underscores, hyphens).
    This behaves more like a dense-ish vector retrieval proxy than pure token match.
    """
    s = _WS_RE.sub(" ", text.lower()).strip()
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]


@dataclass(frozen=True)
class VectorRetriever:
    """Character n-gram TF-IDF cosine similarity (CPU-friendly vector proxy)."""

    docs: list[Document]
    idf: dict[str, float]
    doc_vecs: dict[str, dict[str, float]]
    doc_norms: dict[str, float]
    ngram_n: int = 4

    @classmethod
    def build(cls, docs: list[Document], stats: CorpusStats) -> "VectorRetriever":
        # Note: we intentionally build a separate vector space (char n-grams) from the
        # token DF/IDF used for keyword/BM25. This creates meaningful disagreement
        # between the two strategies in a toy, dependency-free setup.
        n_docs = max(1, len(docs))
        df: dict[str, int] = {}
        per_doc_ngrams: dict[str, list[str]] = {}
        for d in docs:
            grams = _char_ngrams(d.title + " " + d.text, n=4)
            per_doc_ngrams[d.doc_id] = grams
            for g in set(grams):
                df[g] = df.get(g, 0) + 1

        idf: dict[str, float] = {g: math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5)) for g, c in df.items()}

        doc_vecs: dict[str, dict[str, float]] = {}
        doc_norms: dict[str, float] = {}
        for d in docs:
            grams = per_doc_ngrams[d.doc_id]
            tf = term_freq(grams)
            v: dict[str, float] = {}
            for g, c in tf.items():
                if g not in idf:
                    continue
                v[g] = (1.0 + math.log(c)) * idf[g]
            doc_vecs[d.doc_id] = v
            doc_norms[d.doc_id] = _l2norm(v) or 1.0
        return cls(docs=docs, idf=idf, doc_vecs=doc_vecs, doc_norms=doc_norms, ngram_n=4)

    def search(self, query: str, *, k: int = 5) -> list[RetrievalResult]:
        grams = _char_ngrams(query, n=self.ngram_n)
        tf = term_freq(grams)
        q: dict[str, float] = {}
        for g, c in tf.items():
            if g not in self.idf:
                continue
            q[g] = (1.0 + math.log(c)) * self.idf[g]
        qn = _l2norm(q) or 1.0
        scores: list[float] = []
        for d in self.docs:
            dv = self.doc_vecs[d.doc_id]
            dn = self.doc_norms[d.doc_id]
            sim = _dot(q, dv) / (qn * dn)
            scores.append(sim)
        idxs = stable_topk(scores, k)
        return [RetrievalResult(doc=self.docs[i], score=scores[i]) for i in idxs]


@dataclass(frozen=True)
class KeywordRetriever:
    """BM25-like keyword scoring (no positional index; toy but useful)."""

    docs: list[Document]
    stats: CorpusStats
    doc_tfs: dict[str, dict[str, int]]

    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: list[Document], stats: CorpusStats) -> "KeywordRetriever":
        doc_tfs: dict[str, dict[str, int]] = {}
        for d in docs:
            toks = tokenize(d.title + " " + d.text)
            doc_tfs[d.doc_id] = term_freq(toks)
        return cls(docs=docs, stats=stats, doc_tfs=doc_tfs)

    def search(self, query: str, *, k: int = 5) -> list[RetrievalResult]:
        q_toks = tokenize(query)
        q_tf = term_freq(q_toks)
        scores: list[float] = []
        for d in self.docs:
            tf = self.doc_tfs[d.doc_id]
            dl = self.stats.doc_len.get(d.doc_id, 0)
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / (self.stats.avg_dl or 1.0)))
            s = 0.0
            for t in q_tf.keys():
                if t not in self.stats.idf:
                    continue
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                # Classic BM25 term contribution
                s += self.stats.idf[t] * (f * (self.k1 + 1.0)) / (f + denom_norm)
            scores.append(s)
        idxs = stable_topk(scores, k)
        return [RetrievalResult(doc=self.docs[i], score=scores[i]) for i in idxs]


