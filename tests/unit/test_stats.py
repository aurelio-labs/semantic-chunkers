"""Tests for ``semantic_chunkers.stats``: the plotting guard and the statistics.

Plotting lives behind the ``stats`` extra, so matplotlib is absent from the dev
install. These tests stub it in both directions: missing, to check the error
names the right extra and that nothing is encoded before it is raised, and
present, to check a plot is still drawn.
"""

import inspect
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from semantic_router.encoders.openai import OpenAIEncoder

from semantic_chunkers import StatisticalChunker
from semantic_chunkers.chunkers import statistical
from semantic_chunkers.schema import Chunk
from semantic_chunkers.stats import (
    ChunkStatistics,
    chunk_statistics,
    plot_sentence_similarity_scores,
    plot_similarity_scores,
)

ENCODER_NAME = "text-embedding-3-small"

# Two sentences: RegexSplitter only cuts where an uppercase letter follows.
DOCS = ["Doc one about something. Doc two about something."]
SENTENCES = ["Doc one about something.", "Doc two about something."]


@pytest.fixture
def chunker():
    """A ``StatisticalChunker`` whose encoder returns a fixed embedding."""
    encoder = OpenAIEncoder(name=ENCODER_NAME, openai_api_key="a")
    chunker = StatisticalChunker(encoder=encoder)
    chunker.encoder = Mock(side_effect=lambda docs: np.array([[1, 0] for _ in docs]))
    return chunker


@pytest.fixture
def no_matplotlib(monkeypatch):
    """Make ``import matplotlib`` fail, as it does without the stats extra."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)


@pytest.fixture
def fake_pyplot(monkeypatch):
    """Stand a recording stub in for ``matplotlib.pyplot``."""
    plt = MagicMock(name="pyplot")
    plt.subplots.return_value = (MagicMock(name="figure"), MagicMock(name="axes"))
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = plt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return plt


def test_chunk_statistics_str_lists_every_counter():
    stats = ChunkStatistics(
        total_documents=10,
        total_chunks=4,
        chunks_by_threshold=2,
        chunks_by_max_chunk_size=1,
        chunks_by_last_split=1,
        min_token_size=12,
        max_token_size=300,
        chunks_by_similarity_ratio=0.5,
    )

    assert str(stats) == (
        "Chunking Statistics:\n"
        "  - Total Documents: 10\n"
        "  - Total Chunks: 4\n"
        "  - Chunks by Threshold: 2\n"
        "  - Chunks by Max Chunk Size: 1\n"
        "  - Last Chunk: 1\n"
        "  - Minimum Token Size of Chunk: 12\n"
        "  - Maximum Token Size of Chunk: 300\n"
        "  - Similarity Chunk Ratio: 0.50"
    )


def test_chunk_statistics_separates_threshold_from_max_size_and_last_chunk():
    chunks = [
        Chunk(splits=["a"], is_triggered=True, triggered_score=0.1, token_count=120),
        Chunk(splits=["b"], is_triggered=False, token_count=300),
        Chunk(splits=["c"], is_triggered=False, token_count=40),
    ]

    stats = chunk_statistics(docs=["a", "b", "c"], chunks=chunks)

    assert stats.total_documents == 3
    assert stats.total_chunks == 3
    assert stats.chunks_by_threshold == 1
    assert stats.chunks_by_max_chunk_size == 1
    assert stats.chunks_by_last_split == 1
    assert stats.min_token_size == 40
    assert stats.max_token_size == 300
    assert stats.chunks_by_similarity_ratio == pytest.approx(1 / 3)


def test_chunk_statistics_counts_no_last_chunk_when_the_final_chunk_was_triggered():
    chunks = [
        Chunk(splits=["a"], is_triggered=True, triggered_score=0.1, token_count=10)
    ]

    stats = chunk_statistics(docs=["a"], chunks=chunks)

    assert stats.chunks_by_threshold == 1
    assert stats.chunks_by_max_chunk_size == 0
    assert stats.chunks_by_last_split == 0
    assert stats.chunks_by_similarity_ratio == 1.0


def test_chunk_statistics_of_no_chunks_is_zeroed():
    stats = chunk_statistics(docs=[], chunks=[])

    assert stats.total_chunks == 0
    assert stats.chunks_by_threshold == 0
    assert stats.chunks_by_max_chunk_size == 0
    assert stats.chunks_by_last_split == 0
    assert stats.min_token_size == 0
    assert stats.max_token_size == 0
    assert stats.chunks_by_similarity_ratio == 0


def test_plot_similarity_scores_names_the_stats_extra(no_matplotlib):
    with pytest.raises(ImportError, match=r"semantic-chunkers\[stats\]"):
        plot_similarity_scores(
            similarities=[0.9, 0.2],
            split_indices=[2],
            chunks=[Chunk(splits=["a"], token_count=10)],
            calculated_threshold=0.5,
            window_size=5,
        )


def test_plot_sentence_similarity_scores_names_the_stats_extra(no_matplotlib):
    with pytest.raises(ImportError, match=r"semantic-chunkers\[stats\]"):
        plot_sentence_similarity_scores(
            sentences=["a", "b"],
            encode=Mock(),
            threshold=0.5,
            window_size=1,
        )


def test_plot_sentence_similarity_scores_does_not_encode_without_matplotlib(
    no_matplotlib,
):
    encode = Mock()

    with pytest.raises(ImportError):
        plot_sentence_similarity_scores(
            sentences=["a", "b"], encode=encode, threshold=0.5, window_size=1
        )

    encode.assert_not_called()


def test_plot_sentence_similarity_scores_draws_one_score_per_window(fake_pyplot):
    encoded = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    plot_sentence_similarity_scores(
        sentences=["a", "b", "c"],
        encode=lambda sentences: encoded,
        threshold=0.5,
        window_size=1,
    )

    (scores,), _ = fake_pyplot.plot.call_args
    assert len(scores) == 2
    fake_pyplot.show.assert_called_once()


def test_chunker_plot_similarity_scores_names_the_stats_extra(chunker, no_matplotlib):
    with pytest.raises(ImportError, match=r"semantic-chunkers\[stats\]"):
        chunker.plot_similarity_scores(
            similarities=[0.9],
            split_indices=[1],
            chunks=[Chunk(splits=["a"], token_count=10)],
            calculated_threshold=0.5,
        )


def test_chunker_plot_sentence_similarity_scores_names_the_stats_extra(
    chunker, no_matplotlib
):
    with pytest.raises(ImportError, match=r"semantic-chunkers\[stats\]"):
        chunker.plot_sentence_similarity_scores(docs=DOCS, threshold=0.5, window_size=1)

    chunker.encoder.assert_not_called()


def test_chunker_plot_sentence_similarity_scores_encodes_every_sentence(
    chunker, fake_pyplot
):
    chunker.plot_sentence_similarity_scores(docs=DOCS, threshold=0.5, window_size=1)

    (sentences,), _ = chunker.encoder.call_args
    assert sentences == SENTENCES
    fake_pyplot.show.assert_called_once()


def test_chunker_runs_without_matplotlib(no_matplotlib, chunker):
    chunks = chunker(docs=DOCS)

    assert chunks[0], "Expected at least one chunk for the document"


def test_plot_chunks_names_the_stats_extra_without_matplotlib(no_matplotlib, chunker):
    chunker.plot_chunks = True

    with pytest.raises(ImportError, match=r"semantic-chunkers\[stats\]"):
        chunker(docs=DOCS)


def test_plot_chunks_renders_when_matplotlib_is_installed(chunker, fake_pyplot):
    chunker.plot_chunks = True

    chunker(docs=DOCS)

    fake_pyplot.subplots.assert_called_once()
    fake_pyplot.show.assert_called_once()


def test_chunker_records_statistics_for_the_last_batch(no_matplotlib, chunker):
    chunker(docs=DOCS)

    assert isinstance(chunker.statistics, ChunkStatistics)
    assert chunker.statistics.total_documents == len(SENTENCES)
    assert chunker.statistics.total_chunks == 1
    assert chunker.statistics.chunks_by_last_split == 1


def test_statistical_chunker_does_not_import_matplotlib():
    """The core module keeps no trace of the plotting library. VISION.md:42."""
    source = Path(inspect.getsourcefile(statistical) or "").read_text()

    assert "matplotlib" not in source
    assert "pyplot" not in source
