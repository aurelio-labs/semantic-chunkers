"""Fixtures for the functional suite.

These tests chunk a real document with a real encoder — `all-MiniLM-L6-v2`
through sentence-transformers — but every embedding is read from
`data/embeddings.sqlite`, committed next to them. The suite is therefore
deterministic and needs no API key, no model download and no network.

A text that is not in that cache stops the run rather than quietly loading the
model. When the fixture document changes, refill the cache and commit it:

    uv sync --extra dev --extra bench
    SEMANTIC_CHUNKERS_REFRESH_CACHE=1 make test_functional
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.encoders import CachedSentenceTransformerEncoder
from benchmarks.synthetic import load_corpus

CACHE_DIR = Path(__file__).parent / "data"

REFRESH_MESSAGE = (
    "A text is missing from tests/functional/data/embeddings.sqlite, so this "
    "would download all-MiniLM-L6-v2. Refill the cache with "
    "`uv sync --extra dev --extra bench` and "
    "`SEMANTIC_CHUNKERS_REFRESH_CACHE=1 make test_functional`, then commit it."
)


class CachedOnlyEncoder(CachedSentenceTransformerEncoder):
    """The benchmark's cached encoder, kept away from the model.

    Loading the model would tie the suite to a download, so a cache miss is an
    error that says how to refill the cache instead. Set
    `SEMANTIC_CHUNKERS_REFRESH_CACHE=1` to allow the load and write the
    missing embeddings back.

    It also keeps the texts of every request in `requested_batches`, so a test
    can say which text was encoded and how it was batched rather than only how
    many texts there were.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.requested_batches: list[list[str]] = []

    def __call__(self, docs: list[str]) -> list[list[float]]:
        self.requested_batches.append(list(docs))
        return super().__call__(docs)

    def _load(self):
        if os.environ.get("SEMANTIC_CHUNKERS_REFRESH_CACHE") != "1":
            raise RuntimeError(REFRESH_MESSAGE)
        return super()._load()


@pytest.fixture
def encoder() -> CachedOnlyEncoder:
    """`all-MiniLM-L6-v2`, served from the committed embedding cache."""
    return CachedOnlyEncoder()


@pytest.fixture(scope="session")
def document() -> str:
    """Every article introduction in the benchmark corpus, as one document.

    Around two hundred sentences of real prose with real paragraph breaks,
    which is several times the 64 splits a chunker encodes in one batch.
    """
    return "\n\n".join(entry["text"] for entry in load_corpus())
