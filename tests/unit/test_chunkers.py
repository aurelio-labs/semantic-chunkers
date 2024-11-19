from unittest.mock import AsyncMock, Mock, create_autospec

import numpy as np
import pytest
from semantic_router.encoders.base import BaseEncoder
from semantic_router.encoders.openai import OpenAIEncoder

from semantic_chunkers import (
    BaseChunker,
    BaseSplitter,
    ConsecutiveChunker,
    CumulativeChunker,
    StatisticalChunker,
)

ENCODER_NAME = "text-embedding-3-small"


def test_consecutive_sim_splitter():
    # Create a Mock object for the encoder
    mock_encoder = Mock()
    mock_encoder.return_value = np.array([[1, 0], [1, 0.1], [0, 1]])

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        openai_api_key="a",
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
    assert splits[0][0].splits == [
        "doc1 about something"
    ], "First split does not match expected documents"
    assert splits[2][0].splits == [
        "doc3 about something"
    ], "Second split does not match expected documents"


@pytest.mark.asyncio
async def test_async_consecutive_sim_splitter():
    # Create a Mock object for the encoder
    mock_encoder = AsyncMock()

    async def async_return(*args, **kwargs):
        return np.array([[1, 0], [1, 0.1], [0, 1]])

    mock_encoder.acall.side_effect = async_return

    encoder = OpenAIEncoder(
        name=ENCODER_NAME,
        openai_api_key="a",
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
    assert splits[0][0].splits == [
        "doc1 about something"
    ], "First split does not match expected documents"
    assert splits[2][0].splits == [
        "doc3 about something"
    ], "Second split does not match expected documents"


def test_cumulative_sim_splitter():
    # Mock the BaseEncoder
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
        openai_api_key="a",
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
    # Mock the BaseEncoder
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
        openai_api_key="a",
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
    mock_encoder = create_autospec(BaseEncoder)
    # Assuming any return value since it should not reach the point of using the encoder
    mock_encoder.return_value = np.array([[0.5, 0]])

    # TODO JB: this currently doesn't pass, need to fix
    # splitter = ConsecutiveChunker(encoder=mock_encoder, score_threshold=0.5)

    # docs = ["doc1 about something"]
    # chunks = splitter(docs)
    # assert len(chunks) == 1


def test_cumulative_similarity_splitter_single_doc():
    mock_encoder = create_autospec(BaseEncoder)
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
        openai_api_key="a",
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
    assert splits[0][0].splits == [
        "doc1 about something"
    ], "First split does not match expected documents"
    assert splits[2][0].splits == [
        "doc3 about something"
    ], "Second split does not match expected documents"


@pytest.mark.asyncio
async def test_async_statistical_chunker():
    # Create a Mock object for the encoder
    mock_encoder = AsyncMock()
    mock_encoder.side_effect = lambda docs: np.array([[1, 0] for _ in docs])

    encoder = OpenAIEncoder(
        name="",
        openai_api_key="a",
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
    assert splits[0][0].splits == [
        "doc1 about something"
    ], "First split does not match expected documents"
    assert splits[2][0].splits == [
        "doc3 about something"
    ], "Second split does not match expected documents"


@pytest.fixture
def base_splitter_instance():
    # Now MockEncoder includes default values for required fields
    mock_encoder = Mock(spec=BaseEncoder)
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
