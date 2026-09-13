"""``LocalEncoder``: the guard on the extra, and the call it makes.

sentence-transformers is behind the ``local`` extra and absent from the dev
install, so these stub it in both directions — missing, to check the error names
the extra, and present, to check the model is asked for unit vectors and that
the async path returns what the sync path returns.
"""

import asyncio
import sys
from types import ModuleType

import numpy as np
import pytest

from semantic_chunkers.encoders import EncoderError
from semantic_chunkers.encoders.local import LocalEncoder

VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]])


class StubSentenceTransformer:
    """Records how it was built and what it was asked to encode."""

    def __init__(self, name, device=None):
        self.name = name
        self.device = device
        self.calls: list = []

    def encode(self, docs, **kwargs):
        self.calls.append((list(docs), kwargs))
        return VECTORS


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Make ``import sentence_transformers`` fail, as it does without the extra."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_a_missing_install_names_the_extra(no_sentence_transformers):
    with pytest.raises(EncoderError, match=r"semantic-chunkers\[local\]"):
        LocalEncoder()(["a sentence."])


def test_constructing_it_does_not_load_the_model(no_sentence_transformers):
    """The 2 GB of torch is paid on the first call, not on construction."""
    assert LocalEncoder()._model is None


def test_the_model_is_asked_for_normalised_numpy_vectors(fake_sentence_transformers):
    encoder = LocalEncoder(name="all-MiniLM-L6-v2", device="cpu", batch_size=8)

    result = encoder(["one.", "two."])

    docs, kwargs = encoder.model().calls[0]
    assert encoder.model().name == "all-MiniLM-L6-v2"
    assert encoder.model().device == "cpu"
    assert docs == ["one.", "two."]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8
    assert np.array_equal(result, VECTORS)


def test_the_model_is_loaded_once_and_reused(fake_sentence_transformers):
    encoder = LocalEncoder()

    encoder(["one."])
    encoder(["two."])

    assert len(encoder.model().calls) == 2


def test_the_async_path_returns_what_the_sync_path_returns(fake_sentence_transformers):
    encoder = LocalEncoder()

    assert np.array_equal(asyncio.run(encoder.acall(["one."])), encoder(["one."]))
