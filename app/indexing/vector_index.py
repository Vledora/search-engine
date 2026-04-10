"""Semantic vector index using sentence-transformers + FAISS."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indices"
MODEL_NAME = "all-MiniLM-L6-v2"

_model_cache: dict[str, object] = {}


def _load_st_model(model_name: str):
    """Load the model once and cache it across VectorIndex instances."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from sentence_transformers import SentenceTransformer

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*position_ids.*")
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        model = SentenceTransformer(model_name)

    _model_cache[model_name] = model
    return model


class VectorIndex:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self.index = None
        self.doc_ids: list[str] = []

    @property
    def model(self):
        if self._model is None:
            self._model = _load_st_model(self._model_name)
        return self._model

    def build(self, doc_ids: list[str], texts: list[str], batch_size: int = 64) -> None:
        import faiss

        self.doc_ids = doc_ids
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    def save(self, path: Path | None = None) -> None:
        import faiss

        d = path or INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "vector.faiss"))
        with open(d / "vector_ids.json", "w") as f:
            json.dump(self.doc_ids, f)

    def load(self, path: Path | None = None) -> None:
        import faiss

        d = path or INDEX_DIR
        self.index = faiss.read_index(str(d / "vector.faiss"))
        with open(d / "vector_ids.json") as f:
            self.doc_ids = json.load(f)
        self._model = _load_st_model(self._model_name)
