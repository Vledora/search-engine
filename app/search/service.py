"""Unified search service that dispatches to the chosen ranking backend."""

from dataclasses import dataclass

from app.db.models import Document, load_all_documents
from app.text.preprocess import normalize, tokenize
from app.indexing.tfidf_index import TfidfIndex
from app.indexing.bm25_index import Bm25Index
from app.indexing.vector_index import VectorIndex


@dataclass
class SearchResult:
    doc_id: str
    title: str
    url: str
    source: str
    score: float
    snippet: str


class SearchService:
    def __init__(self):
        self.tfidf = TfidfIndex()
        self.bm25 = Bm25Index()
        self.vector = VectorIndex()
        self._docs_by_id: dict[str, Document] = {}
        self._available_methods: set[str] = set()

    def load_indices(self) -> None:
        docs = load_all_documents()
        self._docs_by_id = {d.id: d for d in docs}

        for name, index in [("tfidf", self.tfidf), ("bm25", self.bm25), ("vector", self.vector)]:
            try:
                index.load()
                self._available_methods.add(name)
            except Exception as exc:
                print(f"[warn] Could not load {name} index: {exc}")

    @property
    def available_methods(self) -> list[str]:
        return sorted(self._available_methods)

    def _make_snippet(self, doc: Document, max_chars: int = 250) -> str:
        text = doc.clean_text
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    def _to_results(self, ranked: list[tuple[str, float]]) -> list[SearchResult]:
        results = []
        for doc_id, score in ranked:
            doc = self._docs_by_id.get(doc_id)
            if not doc:
                continue
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    source=doc.source,
                    score=round(score, 4),
                    snippet=self._make_snippet(doc),
                )
            )
        return results

    def search(self, query: str, method: str = "bm25", top_k: int = 10) -> list[SearchResult]:
        if method not in self._available_methods:
            raise ValueError(f"Method '{method}' is not available. Loaded: {self.available_methods}")

        if method == "tfidf":
            clean_q = normalize(query)
            ranked = self.tfidf.search(clean_q, top_k)
        elif method == "bm25":
            tokens = tokenize(normalize(query))
            ranked = self.bm25.search(tokens, top_k)
        elif method == "vector":
            ranked = self.vector.search(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        return self._to_results(ranked)
