"""The chunk-to-source contract: ``content`` is the document, exactly.

Every chunker is driven end to end with a deterministic fake encoder, so what
these assert is what a caller gets back rather than the behaviour of a helper.
The document carries the layout a space-join used to destroy: a blank line,
indentation, a tab, a double space and a trailing newline.
"""

import asyncio
import hashlib
from typing import Any, List

import numpy as np
import pytest
from semantic_router.encoders.base import DenseEncoder

from semantic_chunkers import (
    BaseSplitter,
    ConsecutiveChunker,
    CumulativeChunker,
    RegexChunker,
    RegexSplitter,
    StatisticalChunker,
)
from semantic_chunkers.schema import Chunk

DOC = (
    "Alpha one.\n\n"
    "  Alpha two.\tAlpha three.\n"
    "Beta one.  Beta two.\n\n\n"
    "Gamma one. Gamma two.\t\tGamma three.\n"
)


class TopicEncoder(DenseEncoder):
    """Deterministic stand-in for an embedding model.

    Texts that start with the same word embed identically and anything else is
    far away, so the chunkers cut at the topic changes in ``DOC`` instead of
    returning the whole document as one chunk.
    """

    name: str = "topic-encoder"
    score_threshold: float = 0.45
    type: str = "fake"

    def __call__(self, docs: List[str]) -> List[List[float]]:
        return [self._vector(doc) for doc in docs]

    async def acall(self, docs: List[str]) -> List[List[float]]:
        return self(docs)

    @staticmethod
    def _vector(doc: str) -> List[float]:
        topic = doc.strip().split(" ")[0].lower()
        digest = np.frombuffer(hashlib.sha256(topic.encode()).digest()[:8], np.uint8)
        vec = digest.astype(np.float64) - 128.0
        return list(vec / np.linalg.norm(vec))


class FrameEncoder(DenseEncoder):
    """Stands in for a vision encoder: embeds frames, never text."""

    name: str = "frame-encoder"
    score_threshold: float = 0.45
    type: str = "fake"

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        return [[float(np.mean(frame)), 1.0 - float(np.mean(frame))] for frame in docs]

    async def acall(self, docs: List[Any]) -> List[List[float]]:
        return self(docs)


class ShoutingSplitter(BaseSplitter):
    """A splitter whose splits are not in the document it was given."""

    def __call__(self, doc: str) -> List[str]:
        return [split.upper() for split in RegexSplitter()(doc)]


def chunkers():
    return {
        "regex": RegexChunker(max_chunk_tokens=12),
        "consecutive": ConsecutiveChunker(encoder=TopicEncoder(), score_threshold=0.45),
        "cumulative": CumulativeChunker(encoder=TopicEncoder(), score_threshold=0.45),
        "statistical": StatisticalChunker(
            encoder=TopicEncoder(), min_split_tokens=5, max_split_tokens=20
        ),
    }


@pytest.fixture(params=list(chunkers()))
def chunker(request):
    return chunkers()[request.param]


def test_joining_the_chunks_reproduces_the_document(chunker):
    """The property the space-join could not hold: nothing is lost or invented."""
    chunks = chunker([DOC])[0]

    assert "".join(chunk.content for chunk in chunks) == DOC


def test_every_chunk_is_the_text_at_its_own_offsets(chunker):
    chunks = chunker([DOC])[0]

    for chunk in chunks:
        assert DOC[chunk.start : chunk.end] == chunk.content


def test_offsets_ascend_and_cover_the_document_without_overlapping(chunker):
    chunks = chunker([DOC])[0]

    assert chunks[0].start == 0
    assert chunks[-1].end == len(DOC)
    for left, right in zip(chunks, chunks[1:]):
        assert left.start < left.end == right.start


def test_the_chunkers_cut_this_document_more_than_once(chunker):
    """Guards the tests above: a single chunk would satisfy them trivially."""
    assert len(chunker([DOC])[0]) > 1


def test_content_keeps_the_layout_the_splits_were_stripped_of(chunker):
    """Splits stay stripped; the whitespace between them survives in content."""
    chunks = chunker([DOC])[0]
    joined = "".join(chunk.content for chunk in chunks)

    assert [split for chunk in chunks for split in chunk.splits] == RegexSplitter()(DOC)
    assert "\n\n" in joined and "\t" in joined and "  Alpha two." in joined
    assert all(split == split.strip() for chunk in chunks for split in chunk.splits)


def test_each_chunk_contains_its_own_splits(chunker):
    chunks = chunker([DOC])[0]

    for chunk in chunks:
        for split in chunk.splits:
            assert split in chunk.content


def test_async_chunks_carry_the_same_content_as_sync(chunker):
    """Offsets are one feature across both paths, as VISION.md requires."""
    sync = chunker([DOC])[0]
    asynchronous = asyncio.run(chunker.acall([DOC]))[0]

    assert [(c.start, c.end, c.content) for c in asynchronous] == [
        (c.start, c.end, c.content) for c in sync
    ]


def test_offsets_are_relative_to_each_document_not_the_batch():
    """Two documents in one call each get offsets into themselves."""
    docs = ["Alpha one. Alpha two.", "Beta one. Beta two."]
    chunks_by_doc = RegexChunker(max_chunk_tokens=6)(docs)

    for doc, chunks in zip(docs, chunks_by_doc):
        assert chunks[0].start == 0
        assert chunks[-1].end == len(doc)
        assert "".join(chunk.content for chunk in chunks) == doc


def test_a_chunk_of_frames_has_no_content_or_offsets():
    """The video notebook passes pre-split frames; that path must still work."""
    frames = [np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2))]
    chunker = ConsecutiveChunker(encoder=FrameEncoder(), score_threshold=0.45)

    chunks = chunker([frames])[0]

    assert len(chunks) == 2
    assert [len(chunk.splits) for chunk in chunks] == [2, 2]
    assert all(
        chunk.content is None and chunk.start is None and chunk.end is None
        for chunk in chunks
    )


def test_print_shows_the_document_text_and_survives_a_chunk_without_content(capsys):
    """``print`` reads ``content``; a chunk of frames has none and used to raise."""
    chunker = RegexChunker(max_chunk_tokens=12)
    chunker.print(chunker([DOC])[0])
    assert "Alpha two." in capsys.readouterr().out

    frames = [np.zeros((2, 2)), np.ones((2, 2))]
    frame_chunker = ConsecutiveChunker(encoder=FrameEncoder(), score_threshold=0.45)
    frame_chunker.print(frame_chunker([frames])[0])

    assert "Split 1" in capsys.readouterr().out


def test_a_chunk_built_by_hand_has_no_content():
    """``content`` is text from a document, not the splits joined up."""
    chunk = Chunk(splits=["Alpha one.", "Alpha two."])

    assert chunk.content is None
    assert chunk.start is None and chunk.end is None
    assert Chunk(splits=[], content="Alpha.", start=0, end=6).content == "Alpha."


def test_a_splitter_that_rewrites_its_text_still_chunks_without_offsets():
    """Offsets are dropped rather than guessed when splits are not in the document."""
    chunker = ConsecutiveChunker(
        encoder=TopicEncoder(), splitter=ShoutingSplitter(), score_threshold=0.45
    )

    chunks = chunker([DOC])[0]

    assert [split for chunk in chunks for split in chunk.splits] == ShoutingSplitter()(
        DOC
    )
    assert all(chunk.content is None and chunk.start is None for chunk in chunks)
