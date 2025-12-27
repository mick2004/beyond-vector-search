from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import Document, QueryLabel


def repo_root() -> Path:
    # src/beyond_vector_search/data.py -> repo root
    # parents[0]=beyond_vector_search, [1]=src, [2]=repo_root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def load_corpus(path: Path | None = None) -> list[Document]:
    path = path or (data_dir() / "corpus.jsonl")
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Document(doc_id=obj["doc_id"], title=obj["title"], text=obj["text"]))
    return docs


def load_labels(path: Path | None = None) -> list[QueryLabel]:
    path = path or (data_dir() / "labels.jsonl")
    labels: list[QueryLabel] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels.append(
                QueryLabel(
                    query_id=obj["query_id"],
                    query=obj["query"],
                    expected_doc_id=obj["expected_doc_id"],
                    expected_answer=obj["expected_answer"],
                )
            )
    return labels


