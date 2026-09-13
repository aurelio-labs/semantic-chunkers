"""The encoder protocol: what satisfies it, and what happens when it does not.

The point of the protocol is that nothing has to inherit from us. These tests
stand in for the three kinds of encoder a user brings — a bare callable, an
object shaped like a ``semantic_router`` encoder, and one with no async path —
and check each one against the chunkers rather than against an isinstance call
alone.
"""

import asyncio

import numpy as np
import pytest
from pydantic import BaseModel

from semantic_chunkers import (
    CallableEncoder,
    ConsecutiveChunker,
    DenseEncoder,
    EncoderError,
)

SENTENCES = ["Doc one about something.", "Doc two about something."]
# One document the splitter cuts into those two sentences, so a chunker that
# reads the encoder at all finds two chunks in it.
DOCS = [" ".join(SENTENCES)]


def vectors(docs):
    """One distinct unit vector per document, so every pair is dissimilar."""
    return np.array([[1.0, 0.0] if "one" in doc else [0.0, 1.0] for doc in docs])


class SemanticRouterShapedEncoder(BaseModel):
    """What a ``semantic_router`` encoder looks like from the outside."""

    name: str = "shaped"
    score_threshold: float = 0.5

    def __call__(self, docs):
        return vectors(docs)

    async def acall(self, docs):
        return vectors(docs)


class SyncOnlyEncoder:
    """An encoder whose author never wrote an async path."""

    def __call__(self, docs):
        return vectors(docs)


class StubbedAsyncEncoder:
    """Issue #32: ``acall`` exists but is the base class's unimplemented stub."""

    def __call__(self, docs):
        return vectors(docs)

    async def acall(self, docs):
        raise NotImplementedError("Subclasses must implement this method")


def test_a_semantic_router_shaped_encoder_satisfies_the_protocol():
    assert isinstance(SemanticRouterShapedEncoder(), DenseEncoder)


def test_a_callable_encoder_satisfies_the_protocol():
    assert isinstance(CallableEncoder(vectors), DenseEncoder)


def test_a_bare_callable_does_not_satisfy_the_protocol():
    """It has no ``acall``, which is exactly what CallableEncoder is for."""
    assert not isinstance(vectors, DenseEncoder)


def test_callable_encoder_acall_returns_what_call_returns():
    encoder = CallableEncoder(vectors)

    assert np.array_equal(asyncio.run(encoder.acall(SENTENCES)), encoder(SENTENCES))


def test_callable_encoder_rejects_something_that_is_not_callable():
    with pytest.raises(EncoderError, match="needs a callable"):
        CallableEncoder("text-embedding-3-small")


def test_a_callable_encoder_chunks_the_same_way_sync_and_async():
    chunker = ConsecutiveChunker(encoder=CallableEncoder(vectors), score_threshold=0.9)

    sync_chunks = chunker(DOCS)
    async_chunks = asyncio.run(chunker.acall(DOCS))

    assert [[chunk.splits for chunk in doc] for doc in sync_chunks] == [
        [chunk.splits for chunk in doc] for doc in async_chunks
    ]


def test_a_sync_only_encoder_still_chunks_synchronously():
    chunker = ConsecutiveChunker(encoder=SyncOnlyEncoder(), score_threshold=0.9)

    assert len(chunker(DOCS)[0]) == 2


def test_a_sync_only_encoder_names_itself_when_the_async_path_is_used():
    chunker = ConsecutiveChunker(encoder=SyncOnlyEncoder(), score_threshold=0.9)

    with pytest.raises(EncoderError) as error:
        asyncio.run(chunker.acall(DOCS))

    assert "SyncOnlyEncoder" in str(error.value)
    assert "acall" in str(error.value)


def test_an_unimplemented_acall_names_the_encoder_instead_of_a_third_party_file():
    """Issue #32 surfaced as a NotImplementedError from semantic_router's base."""
    chunker = ConsecutiveChunker(encoder=StubbedAsyncEncoder(), score_threshold=0.9)

    with pytest.raises(EncoderError) as error:
        asyncio.run(chunker.acall(DOCS))

    assert "StubbedAsyncEncoder" in str(error.value)
    assert "CallableEncoder" in str(error.value)


def test_wrapping_a_sync_only_encoder_gives_it_the_async_path():
    """The fix the error message suggests has to actually work."""
    chunker = ConsecutiveChunker(
        encoder=CallableEncoder(SyncOnlyEncoder()), score_threshold=0.9
    )

    assert len(asyncio.run(chunker.acall(DOCS))[0]) == 2


def test_the_heavy_encoders_are_not_imported_by_importing_the_package():
    """``import semantic_chunkers`` must not pay for httpx or torch."""
    import subprocess
    import sys

    script = (
        "import sys, semantic_chunkers;"
        "print(any(m == 'httpx' or m.startswith('torch') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_the_lazily_exported_names_are_reachable():
    import semantic_chunkers

    assert semantic_chunkers.OpenAIEncoder.__name__ == "OpenAIEncoder"
    assert "OpenAIEncoder" in dir(semantic_chunkers)
    with pytest.raises(AttributeError):
        semantic_chunkers.NotAnEncoder
