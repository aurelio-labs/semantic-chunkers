import asyncio
from unittest.mock import AsyncMock, Mock, create_autospec

import numpy as np
import pytest

from semantic_chunkers import (
    BaseChunker,
    BaseSplitter,
    ConsecutiveChunker,
    CumulativeChunker,
    DenseEncoder,
    EncoderError,
    OpenAIEncoder,
    RegexSplitter,
    StatisticalChunker,
)

ENCODER_NAME = "text-embedding-3-small"


def test_consecutive_sim_splitter():
    # Create a Mock object for the encoder
    mock_encoder = Mock()
    mock_encoder.return_value = np.array([[1, 0], [1, 0.1], [0, 1]])

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    # Instantiate the ConsecutiveSimSplitter with the mock encoder
    splitter = ConsecutiveChunker(encoder=encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1 about something", "doc2 about something", "doc3 about something"]

    # Use the splitter to split the documents
    splits = splitter(docs)

    # Verify the splits
    print(splits)
    assert len(splits) == 3, "Expected three sets of chunks"
    assert splits[0][0].splits == ["doc1 about something"], (
        "First split does not match expected documents"
    )
    assert splits[2][0].splits == ["doc3 about something"], (
        "Second split does not match expected documents"
    )


@pytest.mark.asyncio
async def test_async_consecutive_sim_splitter():
    # Create a Mock object for the encoder
    mock_encoder = AsyncMock()

    async def async_return(*args, **kwargs):
        return np.array([[1, 0], [1, 0.1], [0, 1]])

    mock_encoder.acall.side_effect = async_return

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    # Instantiate the ConsecutiveSimSplitter with the mock encoder
    splitter = ConsecutiveChunker(encoder=encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1 about something", "doc2 about something", "doc3 about something"]

    # Use the splitter to split the documents
    splits = await splitter.acall(docs)

    # Verify the splits
    print(splits)
    assert len(splits) == 3, "Expected three sets of chunks"
    assert splits[0][0].splits == ["doc1 about something"], (
        "First split does not match expected documents"
    )
    assert splits[2][0].splits == ["doc3 about something"], (
        "Second split does not match expected documents"
    )


def test_cumulative_sim_splitter():
    # Mock the DenseEncoder
    mock_encoder = Mock()
    # Adjust the side_effect to simulate the encoder's behavior for cumulative document comparisons
    # This simplistic simulation assumes binary embeddings for demonstration purposes
    # Define a side_effect function for the mock encoder
    mock_encoder.side_effect = lambda x: (
        [[0.5, 0]] if "doc1" in x or "doc1\ndoc2" in x or "doc2" in x else [[0, 0.5]]
    )

    # Instantiate the CumulativeSimSplitter with the mock encoder
    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    splitter = CumulativeChunker(encoder=encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = [
        "doc1 about something",
        "doc2 about something",
        "doc3 about something",
        "doc4 about something",
        "doc5 about something",
    ]

    # Use the splitter to split the documents
    splits = splitter(docs)

    # Verify the splits
    # The expected outcome needs to match the logic defined in your mock_encoder's side_effect
    assert len(splits) == 5, f"{len(splits)}"


@pytest.mark.asyncio
async def test_async_cumulative_sim_splitter():
    # Mock the DenseEncoder
    mock_encoder = AsyncMock()
    # Adjust the side_effect to simulate the encoder's behavior for cumulative document comparisons
    # This simplistic simulation assumes binary embeddings for demonstration purposes
    # Define a side_effect function for the mock encoder
    mock_encoder.side_effect = lambda x: (
        [[0.5, 0]] if "doc1" in x or "doc1\ndoc2" in x or "doc2" in x else [[0, 0.5]]
    )

    # Instantiate the CumulativeSimSplitter with the mock encoder
    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    splitter = CumulativeChunker(encoder=encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = [
        "doc1 about something",
        "doc2 about something",
        "doc3 about something",
        "doc4 about something",
        "doc5 about something",
    ]

    # Use the splitter to split the documents
    splits = await splitter.acall(docs)

    # Verify the splits
    # The expected outcome needs to match the logic defined in your mock_encoder's side_effect
    assert len(splits) == 5, f"{len(splits)}"


def test_consecutive_similarity_splitter_single_doc():
    mock_encoder = create_autospec(DenseEncoder)
    # Assuming any return value since it should not reach the point of using the encoder
    mock_encoder.return_value = np.array([[0.5, 0]])

    # TODO JB: this currently doesn't pass, need to fix
    # splitter = ConsecutiveChunker(encoder=mock_encoder, score_threshold=0.5)

    # docs = ["doc1 about something"]
    # chunks = splitter(docs)
    # assert len(chunks) == 1


def test_cumulative_similarity_splitter_single_doc():
    mock_encoder = create_autospec(DenseEncoder)
    # Assuming any return value since it should not reach the point of using the encoder
    mock_encoder.return_value = np.array([[0.5, 0]])

    splitter = CumulativeChunker(encoder=mock_encoder, score_threshold=0.5)

    docs = ["doc1 about something"]
    chunks = splitter(docs)
    assert len(chunks) == 1


def test_statistical_chunker():
    # Create a Mock object for the encoder
    mock_encoder = Mock()
    mock_encoder.side_effect = lambda docs: np.array([[1, 0] for _ in docs])

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    # Instantiate the ConsecutiveSimSplitter with the mock encoder
    splitter = StatisticalChunker(encoder=encoder)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1 about something", "doc2 about something", "doc3 about something"]

    # Use the splitter to split the documents
    splits = splitter(docs=docs)

    # Verify the splits
    print(splits)
    assert len(splits) == 3, "Expected three sets of chunks"
    assert splits[0][0].splits == ["doc1 about something"], (
        "First split does not match expected documents"
    )
    assert splits[2][0].splits == ["doc3 about something"], (
        "Second split does not match expected documents"
    )


@pytest.mark.asyncio
async def test_async_statistical_chunker():
    # Create a Mock object for the encoder
    mock_encoder = AsyncMock()
    # The chunker awaits encoder.acall, so the side effect belongs there; on
    # the mock itself it left every batch encoded as nothing at all.
    mock_encoder.acall.side_effect = lambda docs: np.array([[1, 0] for _ in docs])

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        api_key="a",
    )
    # Instantiate the ConsecutiveSimSplitter with the mock encoder
    splitter = StatisticalChunker(encoder=encoder)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1 about something", "doc2 about something", "doc3 about something"]

    # Use the splitter to split the documents
    splits = await splitter.acall(docs=docs)

    # Verify the splits
    print(splits)
    assert len(splits) == 3, "Expected three sets of chunks"
    assert splits[0][0].splits == ["doc1 about something"], (
        "First split does not match expected documents"
    )
    assert splits[2][0].splits == ["doc3 about something"], (
        "Second split does not match expected documents"
    )


@pytest.mark.asyncio
async def test_async_statistical_chunker_raises_when_the_encoder_times_out():
    """A timed-out encode used to arrive as None and fail as a TypeError.

    The encoder raises the timeout that a stalled one would have raised from
    inside the retry helper, so the test reaches the same branch without
    waiting out the budget.
    """
    mock_encoder = Mock()
    mock_encoder.acall = AsyncMock(side_effect=asyncio.TimeoutError)

    chunker = StatisticalChunker(encoder=OpenAIEncoder(name=ENCODER_NAME, api_key="a"))
    chunker.encoder = mock_encoder

    with pytest.raises(asyncio.TimeoutError):
        await chunker.acall(docs=["doc1 about something. Doc2 about something."])


@pytest.mark.asyncio
async def test_statistical_chunker_rejects_a_document_that_is_not_a_string():
    """The chunker counted the tokens of a document before checking its type.

    Anything but a string died inside tiktoken as a TypeError about PyString,
    with the ValueError the chunker meant to raise never reached.
    """
    mock_encoder = Mock()
    mock_encoder.side_effect = lambda docs: np.array([[1, 0] for _ in docs])

    chunker = StatisticalChunker(encoder=OpenAIEncoder(name=ENCODER_NAME, api_key="a"))
    chunker.encoder = mock_encoder

    with pytest.raises(ValueError, match="must be a string"):
        chunker(docs=[["a frame", "another frame"]])

    with pytest.raises(ValueError, match="must be a string"):
        await chunker.acall(docs=[["a frame", "another frame"]])


@pytest.fixture
def base_splitter_instance():
    # Now MockEncoder includes default values for required fields
    mock_encoder = Mock(spec=DenseEncoder)
    mock_encoder.name = "mock_encoder"
    mock_encoder.score_threshold = 0.5
    mock_splitter = Mock(spec=BaseSplitter)
    return BaseChunker(
        name="test_splitter",
        encoder=mock_encoder,
        splitter=mock_splitter,
    )


def test_base_splitter_call_not_implemented(base_splitter_instance):
    with pytest.raises(NotImplementedError):
        base_splitter_instance(["document"])


def test_base_chunker_leaves_the_encoder_unset_when_omitted():
    """A chunker that needs no encoder gets none, not one that cannot encode."""
    chunker = BaseChunker(name="t", splitter=RegexSplitter())
    assert chunker.encoder is None


def test_base_chunker_rejects_an_encoder_that_cannot_be_called():
    """A model name where an encoder belongs is caught at construction."""
    with pytest.raises(EncoderError, match="must be callable"):
        BaseChunker(
            name="t", encoder="text-embedding-3-small", splitter=RegexSplitter()
        )


def test_regex_splitter_accepts_custom_pattern():
    splitter = RegexSplitter(regex_pattern=r"\|")
    assert splitter.regex_pattern == r"\|"
    assert splitter("a|b|c") == ["a", "b", "c"]


class WordSplitter(BaseSplitter):
    """A splitter that implements nothing but ``__call__``."""

    def __call__(self, doc: str) -> list[str]:
        return [word for word in doc.split(" ") if word]


class LoudSplitter(WordSplitter):
    """A splitter whose splits cannot be found in the document."""

    def __call__(self, doc: str) -> list[str]:
        return [word.upper() for word in super().__call__(doc)]


def test_base_splitter_locates_the_splits_of_a_custom_splitter():
    """A splitter written before offsets existed still gets them."""
    doc = "alpha  beta gamma"

    spans = WordSplitter().spans(doc)

    assert [doc[start:end] for start, end in spans] == ["alpha", "beta", "gamma"]
    assert spans == [(0, 5), (7, 11), (12, 17)]


def test_base_splitter_gives_no_spans_when_the_text_was_rewritten():
    """Offsets that might be wrong are worse than none, so none are returned."""
    assert LoudSplitter().spans("alpha beta") == []


def test_base_splitter_gives_no_spans_for_an_empty_document():
    assert WordSplitter().spans("") == []
