"""BM25 index using rank_bm25."""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indices"


class Bm25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []

    def build(self, doc_ids: list[str], tokenized_docs: list[list[str]]) -> None:
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: Path | None = None) -> None:
        d = path or INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)

    def load(self, path: Path | None = None) -> None:
        d = path or INDEX_DIR
        with open(d / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
