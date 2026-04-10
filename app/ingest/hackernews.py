"""Fetch stories and comments from the public Hacker News Firebase API."""

import sys

import httpx

from app.db.models import Document
from app.text.preprocess import preprocess

BASE_URL = "https://hacker-news.firebaseio.com/v0"


def _get_json(client: httpx.Client, path: str) -> dict | list | None:
    resp = client.get(f"{BASE_URL}/{path}.json", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_top_story_ids(client: httpx.Client, limit: int = 200) -> list[int]:
    ids = _get_json(client, "topstories") or []
    return ids[:limit]


def fetch_best_story_ids(client: httpx.Client, limit: int = 200) -> list[int]:
    ids = _get_json(client, "beststories") or []
    return ids[:limit]


def ingest(max_stories: int = 400) -> list[Document]:
    """Fetch top + best stories and convert them into Documents."""
    docs: list[Document] = []
    seen: set[str] = set()

    with httpx.Client(http2=True) as client:
        top_ids = fetch_top_story_ids(client, max_stories // 2)
        best_ids = fetch_best_story_ids(client, max_stories // 2)
        all_ids = list(dict.fromkeys(top_ids + best_ids))[:max_stories]

        for i, sid in enumerate(all_ids):
            if i % 50 == 0:
                print(f"       [{i}/{len(all_ids)}] fetching stories ...", flush=True)

            try:
                item = _get_json(client, f"item/{sid}")
            except Exception:
                continue
            if not item or item.get("dead") or item.get("deleted"):
                continue

            doc_id = f"hn_{item['id']}"
            if doc_id in seen:
                continue
            seen.add(doc_id)

            title = item.get("title", "")
            raw_html = item.get("text", "") or title
            url = item.get("url", f"https://news.ycombinator.com/item?id={item['id']}")

            combined_html = f"<h1>{title}</h1>\n{raw_html}"
            clean_text, tokens = preprocess(combined_html)
            if len(tokens) < 3:
                continue

            docs.append(
                Document(
                    id=doc_id,
                    source="hackernews",
                    title=title,
                    url=url,
                    raw_html=combined_html,
                    clean_text=clean_text,
                    tokens=tokens,
                )
            )

    print(f"       fetched {len(docs)} stories total", flush=True)
    return docs
