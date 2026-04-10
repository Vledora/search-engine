"""Fetch articles from the public Wikipedia API."""

import sys

import httpx

from app.db.models import Document
from app.text.preprocess import preprocess

SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
PARSE_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "SearchEngineMVP/1.0 (portfolio project; contact@example.com)",
}


def search_titles(client: httpx.Client, query: str, limit: int = 20) -> list[dict]:
    """Return page titles + keys matching *query*."""
    resp = client.get(
        SEARCH_URL,
        params={"q": query, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("pages", [])


def fetch_page_html(client: httpx.Client, title: str) -> str:
    """Return the parsed HTML of a Wikipedia article."""
    resp = client.get(
        PARSE_URL,
        params={
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("parse", {}).get("text", "")


def ingest(seed_queries: list[str], pages_per_query: int = 20) -> list[Document]:
    """Discover and fetch Wikipedia articles for each seed query."""
    seen_ids: set[str] = set()
    docs: list[Document] = []

    with httpx.Client(headers=HEADERS, http2=True) as client:
        for qi, query in enumerate(seed_queries, 1):
            print(f"       [{qi}/{len(seed_queries)}] query: {query!r}", flush=True)
            try:
                pages = search_titles(client, query, limit=pages_per_query)
            except Exception as exc:
                print(f"         search failed: {exc}", file=sys.stderr)
                continue

            for pi, page in enumerate(pages):
                page_id = f"wiki_{page['id']}"
                if page_id in seen_ids:
                    continue
                seen_ids.add(page_id)

                title = page.get("title", "")
                try:
                    raw_html = fetch_page_html(client, title)
                except Exception:
                    continue

                clean_text, tokens = preprocess(raw_html)
                if len(tokens) < 10:
                    continue

                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                docs.append(
                    Document(
                        id=page_id,
                        source="wikipedia",
                        title=title,
                        url=url,
                        raw_html=raw_html,
                        clean_text=clean_text,
                        tokens=tokens,
                    )
                )

            print(f"         fetched {len(pages)} pages, {len(docs)} total docs so far", flush=True)

    return docs
