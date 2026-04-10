"""TF-IDF index built with scikit-learn. Supports cosine-similarity search."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indices"


class TfidfIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=30_000,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = None
        self.doc_ids: list[str] = []

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: Path | None = None) -> None:
        d = path or INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "tfidf.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "doc_ids": self.doc_ids}, f)

    def load(self, path: Path | None = None) -> None:
        d = path or INDEX_DIR
        with open(d / "tfidf.pkl", "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.doc_ids = data["doc_ids"]
