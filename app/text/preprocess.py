"""Deterministic text preprocessing: HTML strip, lowercase, stop-word removal, tokenization."""

import re
from functools import lru_cache

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_data() -> None:
    for resource in ("punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = word_tokenize(text)
    if remove_stopwords:
        stops = _stop_words()
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return tokens


def preprocess(raw_html: str) -> tuple[str, list[str]]:
    """Full pipeline: HTML -> clean text + token list."""
    plain = strip_html(raw_html)
    clean = normalize(plain)
    tokens = tokenize(clean)
    return clean, tokens
