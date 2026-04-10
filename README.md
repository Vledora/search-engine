# Search Engine MVP

A portfolio-ready mini search engine that ingests content from public APIs (Wikipedia, Hacker News), preprocesses text, builds multiple retrieval indices, and serves a web UI to compare ranking methods side-by-side.

## Architecture

```
Public APIs ─► Ingestion Pipeline ─► HTML Strip & Normalize ─► Tokenize / Remove Stop Words
                                                                    │
                                          ┌─────────────────────────┼─────────────────────────┐
                                          ▼                         ▼                         ▼
                                     TF-IDF Index            BM25 Index              Vector Index
                                     (scikit-learn)          (rank-bm25)        (sentence-transformers
                                                                                      + FAISS)
                                          └─────────────────────────┼─────────────────────────┘
                                                                    ▼
                                                          FastAPI Search API
                                                                    ▼
                                                              Search UI
```

## How it works

| Stage | What happens |
|---|---|
| **Ingestion** | Wikipedia articles are fetched via the MediaWiki REST API; Hacker News stories via the Firebase API. Each item is normalized into a shared `Document` schema. |
| **Preprocessing** | Raw HTML is stripped with BeautifulSoup, lowercased, cleaned of punctuation, stop-words are removed, and tokens are generated with NLTK. |
| **TF-IDF** | `TfidfVectorizer` from scikit-learn builds a sparse term-document matrix. Queries are ranked by cosine similarity. |
| **BM25** | `BM25Okapi` from rank-bm25 scores documents using the Okapi BM25 probabilistic ranking function with term frequency saturation and document length normalization. |
| **Vector search** | Documents are encoded into dense embeddings with `all-MiniLM-L6-v2` via sentence-transformers. FAISS `IndexFlatIP` stores and searches them by inner product (cosine similarity on normalized vectors). |
| **Web UI** | A FastAPI app exposes `/api/search` and a Jinja2-rendered UI where users pick a method (or compare all three side-by-side). |

## Quick start

```bash
# Clone and enter the project
cd search-engine

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ingest documents from Wikipedia + Hacker News and build indices
python scripts/reindex.py

# Start the web app
uvicorn app.main:app --reload
# Open http://localhost:8000
```

If you don't have network access (or just want a quick demo), use the sample data seeder instead:

```bash
python scripts/seed_sample_data.py
uvicorn app.main:app --reload
```

## Project structure

```
search-engine/
├── app/
│   ├── main.py              # FastAPI routes and app startup
│   ├── db/
│   │   └── models.py        # Document dataclass + SQLite storage
│   ├── ingest/
│   │   ├── wikipedia.py     # Wikipedia API connector
│   │   └── hackernews.py    # Hacker News API connector
│   ├── text/
│   │   └── preprocess.py    # HTML strip, lowercase, stop-word removal, tokenization
│   ├── indexing/
│   │   ├── tfidf_index.py   # TF-IDF build + cosine search
│   │   ├── bm25_index.py    # BM25 build + search
│   │   └── vector_index.py  # Embedding + FAISS search
│   └── search/
│       └── service.py       # Unified search dispatcher
├── templates/
│   └── index.html           # Search UI (dark theme, method tabs, comparison view)
├── scripts/
│   ├── reindex.py           # Full ingest + index pipeline
│   └── seed_sample_data.py  # Offline sample data for quick demo
├── data/                    # SQLite DB + index files (gitignored)
├── requirements.txt
└── README.md
```

## Ranking methods compared

| Method | Type | Strengths | Weaknesses |
|---|---|---|---|
| **TF-IDF** | Lexical (sparse) | Fast, interpretable, no training needed | No semantic understanding; exact keyword match only |
| **BM25** | Lexical (sparse) | Better than TF-IDF via length normalization and TF saturation; industry standard baseline | Still keyword-dependent; misses synonyms |
| **Vector** | Semantic (dense) | Understands meaning and synonyms; handles natural language queries | Slower to build; requires model download; less precise on exact keyword matches |

Try these queries to see the differences:

- `"machine learning"` — BM25 nails exact keyword matches
- `"how do computers understand language"` — vector search surfaces NLP articles despite no keyword overlap
- `"python web development"` — TF-IDF and BM25 differ on which signals to weight

## API reference

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/search` | GET | `q` (required), `method` (bm25\|tfidf\|vector), `top_k` (1-50) | Returns ranked search results |
| `/api/methods` | GET | — | Lists currently available ranking methods |
| `/` | GET | — | Serves the search UI |

## Tech stack

- **Python 3.11+** with FastAPI + Uvicorn
- **BeautifulSoup** + **NLTK** for text preprocessing
- **scikit-learn** for TF-IDF
- **rank-bm25** for BM25
- **sentence-transformers** + **FAISS** for vector search
- **SQLite** for document storage
- **Jinja2** for HTML templating

## Resume positioning

> Built an end-to-end search engine over public web content (Wikipedia, Hacker News) using API-based crawling and text normalization. Implemented and compared TF-IDF, BM25, and semantic vector retrieval, exposing a FastAPI web app for querying and visualizing ranked results across methods.
