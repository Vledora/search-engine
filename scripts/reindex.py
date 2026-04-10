#!/usr/bin/env python3
"""Ingest documents from public APIs and build all search indices."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import upsert_documents, load_all_documents, count_documents
from app.ingest import wikipedia, hackernews
from app.indexing.tfidf_index import TfidfIndex
from app.indexing.bm25_index import Bm25Index
from app.indexing.vector_index import VectorIndex

WIKI_SEEDS = [
    "artificial intelligence",
    "machine learning",
    "python programming language",
    "web search engine",
    "information retrieval",
    "natural language processing",
    "deep learning",
    "computer science",
    "data structure",
    "algorithm",
    "operating system",
    "database management",
    "cloud computing",
    "cybersecurity",
    "blockchain technology",
]


def main() -> None:
    print("=== Search Engine Reindex ===\n")

    # --- Ingest Wikipedia ---
    print(f"[1/5] Fetching Wikipedia articles for {len(WIKI_SEEDS)} seed queries ...")
    t0 = time.time()
    wiki_docs = wikipedia.ingest(WIKI_SEEDS, pages_per_query=20)
    print(f"       Got {len(wiki_docs)} Wikipedia documents in {time.time() - t0:.1f}s")

    # --- Ingest Hacker News ---
    print("[2/5] Fetching Hacker News stories ...")
    t0 = time.time()
    hn_docs = hackernews.ingest(max_stories=400)
    print(f"       Got {len(hn_docs)} Hacker News documents in {time.time() - t0:.1f}s")

    # --- Store ---
    all_docs = wiki_docs + hn_docs
    upsert_documents(all_docs)
    print(f"\n       Total documents in DB: {count_documents()}")

    # --- Reload to guarantee consistent ordering ---
    docs = load_all_documents()
    doc_ids = [d.id for d in docs]
    clean_texts = [d.clean_text for d in docs]
    token_lists = [d.tokens for d in docs]

    # --- Build TF-IDF ---
    print("\n[3/5] Building TF-IDF index ...")
    t0 = time.time()
    tfidf = TfidfIndex()
    tfidf.build(doc_ids, clean_texts)
    tfidf.save()
    print(f"       Done in {time.time() - t0:.1f}s")

    # --- Build BM25 ---
    print("[4/5] Building BM25 index ...")
    t0 = time.time()
    bm25 = Bm25Index()
    bm25.build(doc_ids, token_lists)
    bm25.save()
    print(f"       Done in {time.time() - t0:.1f}s")

    # --- Build Vector index ---
    print("[5/5] Building vector index (this may take a minute) ...")
    t0 = time.time()
    try:
        vec = VectorIndex()
        vec.build(doc_ids, clean_texts)
        vec.save()
        print(f"       Done in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"       Skipped vector index: {exc}")

    print(f"\n=== Indexing complete. {len(docs)} documents indexed. ===")


if __name__ == "__main__":
    main()
