"""Each chunker must cut a document the same way whether it is called or awaited.

The document runs to some two hundred sentences, well past the 64 splits the
statistical chunker encodes in one batch, because the batch boundary is where
the two paths diverged: the sync path carried an unfinished chunk into the next
batch and the async path started afresh, so every 64th split ended a chunk.
"""

from math import ceil

import pytest

from semantic_chunkers import (
    ConsecutiveChunker,
    CumulativeChunker,
    RegexChunker,
    RegexSplitter,
    StatisticalChunker,
)

BATCH_SIZE = 64  # the default every chunker encodes with


@pytest.fixture(params=["statistical", "consecutive", "cumulative", "regex"])
def chunker(request, encoder):
    """Each chunker in turn, on the cached local encoder."""
    if request.param == "statistical":
        return StatisticalChunker(encoder=encoder)
    if request.param == "consecutive":
        return ConsecutiveChunker(encoder=encoder)
    if request.param == "cumulative":
        return CumulativeChunker(encoder=encoder)
    return RegexChunker()


def cuts(chunks):
    """What a chunker decided: where it cut, and what it says about each cut."""
    return [
        (chunk.splits, chunk.is_triggered, chunk.token_count, chunk.start, chunk.end)
        for chunk in chunks
    ]


def scores(chunks):
    """The similarity that triggered each cut, of the chunks that were cut."""
    return [
        chunk.triggered_score for chunk in chunks if chunk.triggered_score is not None
    ]


def test_the_document_is_longer_than_one_encoder_batch(document):
    """Parity below the batch size would prove nothing about batching."""
    assert len(RegexSplitter()(document)) > 2 * BATCH_SIZE


@pytest.mark.asyncio
async def test_chunkers_chunk_the_same_sync_and_async(chunker, document):
    sync_chunks = chunker([document])[0]
    async_chunks = (await chunker.acall([document]))[0]

    # The sizes first: a mismatch there reads better than two hundred chunks
    # of pydantic repr.
    assert [len(chunk.splits) for chunk in sync_chunks] == [
        len(chunk.splits) for chunk in async_chunks
    ]
    assert cuts(sync_chunks) == cuts(async_chunks)
    assert scores(sync_chunks) == pytest.approx(scores(async_chunks))


@pytest.mark.asyncio
async def test_statistical_chunking_encodes_each_split_once(encoder, document):
    """Parity of chunks is not parity of cost: see priority 2 in VISION.md.

    Both paths encode the whole document up front and slice, so a split
    carried into the next batch keeps the embedding it already has instead of
    being paid for twice. Each asks for exactly one text per split — an
    invariant the chunk comparison above would not notice being lost.
    """
    splits = RegexSplitter()(document)
    chunker = StatisticalChunker(encoder=encoder)

    chunker([document])

    assert encoder.requested_texts == len(splits)
    assert encoder.requests == ceil(len(splits) / BATCH_SIZE)

    encoder.reset_counters()
    await chunker.acall([document])

    assert encoder.requested_texts == len(splits)
    assert encoder.requests == ceil(len(splits) / BATCH_SIZE)


@pytest.mark.asyncio
async def test_statistical_chunking_batches_across_documents(encoder, documents):
    """Short documents share an encoder batch: see priority 2 in VISION.md.

    One request per document is one round trip per document against a remote
    encoder, which is what a hundred short documents used to cost. Batches are
    filled from the splits of every document instead, and each document is
    still cut the way it is cut on its own.
    """
    splits = [split for doc in documents for split in RegexSplitter()(doc)]
    chunker = StatisticalChunker(encoder=encoder)

    alone = [chunker([doc])[0] for doc in documents]

    encoder.reset_counters()
    together = chunker(documents)

    assert [cuts(chunks) for chunks in together] == [cuts(chunks) for chunks in alone]
    assert encoder.requested_texts == len(splits)
    assert encoder.requests == ceil(len(splits) / BATCH_SIZE)

    encoder.reset_counters()
    awaited = await chunker.acall(documents)

    assert [cuts(chunks) for chunks in awaited] == [cuts(chunks) for chunks in alone]
    assert encoder.requested_texts == len(splits)
    assert encoder.requests == ceil(len(splits) / BATCH_SIZE)


@pytest.mark.asyncio
async def test_statistical_chunking_a_bare_list_of_splits_cuts_the_same(
    encoder, document
):
    """`_chunk` chunks splits a caller already has, and it must not drift.

    It encodes what it was given, where the document path encodes across every
    document in the call and hands each one its own slice. The two arrive at
    the embeddings differently, so this is the assertion that keeps the chunks
    they produce the same.
    """
    chunker = StatisticalChunker(encoder=encoder)
    splits = RegexSplitter()(document)

    expected = [chunk.splits for chunk in chunker([document])[0]]

    assert [chunk.splits for chunk in chunker._chunk(splits)] == expected
    assert [chunk.splits for chunk in await chunker._async_chunk(splits)] == expected


@pytest.mark.asyncio
async def test_chunks_join_back_into_the_document(chunker, document):
    """Chunking is lossless, both ways round: see VISION.md."""
    for chunks in (chunker([document])[0], (await chunker.acall([document]))[0]):
        assert "".join(chunk.content for chunk in chunks) == document
