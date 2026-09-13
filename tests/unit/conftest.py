"""Shared fixtures for the benchmark tests.

The cached encoder loads its model lazily, so a test can stand a stub in its
place and exercise the cache, the counters and the runner with no
sentence-transformers import and no model download.
"""

import hashlib

import numpy as np
import pytest

from benchmarks.encoders import CachedSentenceTransformerEncoder


class StubModel:
    """Stands in for a ``SentenceTransformer`` and records what reached it.

    Each call is recorded with the cache namespace in force at the time, which
    is what makes "this text was embedded again under a new partition" visible
    to a test. Vectors are a deterministic function of the text, so a cache hit
    and a miss are indistinguishable to the caller.
    """

    def __init__(self, encoder: CachedSentenceTransformerEncoder):
        self._encoder = encoder
        self.calls: list[tuple[str, list[str]]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((self._encoder.namespace, list(texts)))
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        digest = np.frombuffer(hashlib.sha256(text.encode()).digest()[:8], np.uint8)
        vec = digest.astype(np.float32) - 128.0
        return vec / np.linalg.norm(vec)

    @property
    def namespaces(self) -> list[str]:
        """The namespace of every call, in order."""
        return [namespace for namespace, _ in self.calls]

    @property
    def texts(self) -> list[str]:
        """Every text the model embedded, in order."""
        return [text for _, batch in self.calls for text in batch]


@pytest.fixture
def fake_encoder(tmp_path):
    """Build a real cached encoder whose model is a :class:`StubModel`."""

    def build(namespace: str = "") -> CachedSentenceTransformerEncoder:
        encoder = CachedSentenceTransformerEncoder(
            name="stub", cache_dir=tmp_path / "cache", namespace=namespace
        )
        encoder._model = StubModel(encoder)
        return encoder

    return build
