"""Document schema and SQLite storage layer."""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "documents.db"


@dataclass
class Document:
    id: str
    source: str  # "wikipedia" or "hackernews"
    title: str
    url: str
    raw_html: str
    clean_text: str = ""
    tokens: list[str] = field(default_factory=list)

    def to_row(self) -> tuple:
        return (
            self.id,
            self.source,
            self.title,
            self.url,
            self.raw_html,
            self.clean_text,
            json.dumps(self.tokens),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Document":
        return cls(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            url=row["url"],
            raw_html=row["raw_html"],
            clean_text=row["clean_text"],
            tokens=json.loads(row["tokens"]),
        )


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    title       TEXT NOT NULL,
    url         TEXT NOT NULL,
    raw_html    TEXT NOT NULL,
    clean_text  TEXT NOT NULL DEFAULT '',
    tokens      TEXT NOT NULL DEFAULT '[]'
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def upsert_documents(docs: list[Document], db_path: Optional[Path] = None) -> int:
    """Insert or replace documents. Returns number of rows affected."""
    conn = _connect(db_path)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO documents
               (id, source, title, url, raw_html, clean_text, tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [d.to_row() for d in docs],
        )
        conn.commit()
        return len(docs)
    finally:
        conn.close()


def load_all_documents(db_path: Optional[Path] = None) -> list[Document]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM documents").fetchall()
        return [Document.from_row(r) for r in rows]
    finally:
        conn.close()


def count_documents(db_path: Optional[Path] = None) -> int:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()
