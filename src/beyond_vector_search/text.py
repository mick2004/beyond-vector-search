from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """
    Lowercase tokenization tuned for engineering text:
    - keeps hyphen/underscore joined tokens: 'inc-49217', 'user_id'
    - strips punctuation otherwise
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


@dataclass(frozen=True)
class QueryFeatures:
    n_tokens: int
    digit_ratio: float
    oov_ratio: float
    rare_ratio: float


def featurize_query(query: str, *, vocab: set[str], rare_terms: set[str]) -> QueryFeatures:
    toks = tokenize(query)
    n = len(toks)
    if n == 0:
        return QueryFeatures(n_tokens=0, digit_ratio=0.0, oov_ratio=0.0, rare_ratio=0.0)

    digit_ratio = sum(1 for t in toks if any(ch.isdigit() for ch in t)) / n
    oov_ratio = sum(1 for t in toks if t not in vocab) / n
    rare_ratio = sum(1 for t in toks if t in rare_terms) / n
    return QueryFeatures(n_tokens=n, digit_ratio=digit_ratio, oov_ratio=oov_ratio, rare_ratio=rare_ratio)


def join_top_sentences(text: str, *, max_sentences: int = 2) -> str:
    parts = [p.strip() for p in re.split(r"[.!?]\s+", text) if p.strip()]
    if not parts:
        return ""
    out = ". ".join(parts[:max_sentences]).strip()
    return out if out.endswith((".", "!", "?")) else out + "."


def stable_topk(scores: Sequence[float], k: int) -> list[int]:
    """Return indices of top-k scores, stable on ties by index."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


