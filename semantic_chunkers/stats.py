"""Statistics and charts for inspecting how a chunker decided where to cut.

Nothing here decides where a document is cut, so the chunkers can delegate to
it and stay free of chart formatting. matplotlib is not a core dependency: it
comes with `pip install semantic-chunkers[stats]` and is imported only when a
plot is actually drawn.
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from semantic_chunkers.schema import Chunk

__all__ = [
    "ChunkStatistics",
    "chunk_statistics",
    "plot_sentence_similarity_scores",
    "plot_similarity_scores",
]


@dataclass
class ChunkStatistics:
    """Counts describing one batch of chunks, printed by ``enable_statistics``."""

    total_documents: int
    total_chunks: int
    chunks_by_threshold: int
    chunks_by_max_chunk_size: int
    chunks_by_last_split: int
    min_token_size: int
    max_token_size: int
    chunks_by_similarity_ratio: float

    def __str__(self):
        return (
            f"Chunking Statistics:\n"
            f"  - Total Documents: {self.total_documents}\n"
            f"  - Total Chunks: {self.total_chunks}\n"
            f"  - Chunks by Threshold: {self.chunks_by_threshold}\n"
            f"  - Chunks by Max Chunk Size: {self.chunks_by_max_chunk_size}\n"
            f"  - Last Chunk: {self.chunks_by_last_split}\n"
            f"  - Minimum Token Size of Chunk: {self.min_token_size}\n"
            f"  - Maximum Token Size of Chunk: {self.max_token_size}\n"
            f"  - Similarity Chunk Ratio: {self.chunks_by_similarity_ratio:.2f}"
        )


def chunk_statistics(docs: List[str], chunks: List[Chunk]) -> ChunkStatistics:
    """Describe the chunks a batch of documents produced.

    :param docs: The documents that were chunked.
    :param chunks: The chunks they produced, in order.

    :return: The statistics for this batch.
    """
    # Why a chunk was cut is already on the chunk, so read the counts back off
    # the result instead of tallying them inside the chunking loop.
    total = len(chunks)
    # A chunk cut by the similarity threshold carries is_triggered.
    by_threshold = sum(1 for chunk in chunks if chunk.is_triggered)
    # A trailing chunk that was not triggered is whatever the documents left over.
    by_last_split = 1 if chunks and not chunks[-1].is_triggered else 0
    # Everything else was cut by the token limit.
    by_max_size = total - by_threshold - by_last_split
    token_counts = [c.token_count for c in chunks if c.token_count is not None]
    return ChunkStatistics(
        total_documents=len(docs),
        total_chunks=total,
        chunks_by_threshold=by_threshold,
        chunks_by_max_chunk_size=by_max_size,
        chunks_by_last_split=by_last_split,
        min_token_size=min(token_counts, default=0),
        max_token_size=max(token_counts, default=0),
        chunks_by_similarity_ratio=by_threshold / total if total else 0,
    )


def _pyplot():
    """Return ``matplotlib.pyplot``, or say which extra installs it.

    :raises ImportError: When matplotlib is not installed.
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib, which is not installed by default. "
            "Install it with `pip install semantic-chunkers[stats]`."
        ) from e
    return plt


def plot_similarity_scores(
    similarities: List[float],
    split_indices: List[int],
    chunks: List[Chunk],
    calculated_threshold: float,
    window_size: int,
) -> None:
    """Plot the rolling-window similarity of each split, then each chunk's size.

    :param similarities: Similarity score of each split against its window.
    :param split_indices: Indices the chunker decided to cut after.
    :param chunks: The chunks those cuts produced.
    :param calculated_threshold: Threshold the scores were compared against.
    :param window_size: Number of preceding splits in the rolling window.

    :raises ImportError: When matplotlib is not installed.
    """
    plt = _pyplot()
    _, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Upper axes: the score of every split, with each cut and the threshold marked.
    axs[0].plot(similarities, label="Similarity Scores", marker="o")
    for split_index in split_indices:
        axs[0].axvline(
            x=split_index - 1,
            color="r",
            linestyle="--",
            label="Chunk" if split_index == split_indices[0] else "",
        )
    axs[0].axhline(
        y=calculated_threshold,
        color="g",
        linestyle="-.",
        label="Threshold Similarity Score",
    )
    for i, score in enumerate(similarities):
        axs[0].annotate(
            f"{score:.2f}",
            (i, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    axs[0].set_xlabel("Document Segment Index")
    axs[0].set_ylabel("Similarity Score")
    axs[0].set_title(
        f"Threshold: {calculated_threshold} | Window Size: {window_size}",
        loc="right",
        fontsize=10,
    )
    axs[0].legend()

    # Lower axes: the token count of each chunk those cuts produced.
    token_counts = [chunk.token_count for chunk in chunks]
    axs[1].bar(range(len(token_counts)), token_counts, color="lightblue")
    axs[1].set_title("Chunk Token Sizes")
    axs[1].set_xlabel("Chunk Index")
    axs[1].set_ylabel("Token Count")
    axs[1].set_xticks(range(len(token_counts)))
    axs[1].set_xticklabels([str(i) for i in range(len(token_counts))])
    axs[1].grid(True)
    for idx, token_count in enumerate(token_counts):
        if token_count:
            axs[1].text(
                idx, token_count + 0.01, str(token_count), ha="center", va="bottom"
            )

    plt.tight_layout()
    plt.show()


def plot_sentence_similarity_scores(
    sentences: List[str],
    encode: Callable[[List[str]], np.ndarray],
    threshold: float,
    window_size: int,
) -> None:
    """Plot how each sentence compares with the ``window_size`` sentences before it.

    Prints the first sentence after every score below ``threshold``.

    :param sentences: The sentences to compare, in document order.
    :param encode: Callable turning those sentences into embeddings. It is
        called only once matplotlib is known to be importable, so a missing
        install costs no encoder calls.
    :param threshold: Score below which a sentence is reported as a boundary.
    :param window_size: Number of preceding sentences averaged into the window.

    :raises ImportError: When matplotlib is not installed.
    """
    plt = _pyplot()
    encoded_sentences = encode(sentences)

    # Cosine similarity between each sentence and the average of its window.
    similarity_scores = []
    for i in range(window_size, len(encoded_sentences)):
        window_avg_encoding = np.mean(encoded_sentences[i - window_size : i], axis=0)
        sim_score = np.dot(window_avg_encoding, encoded_sentences[i]) / (
            np.linalg.norm(window_avg_encoding) * np.linalg.norm(encoded_sentences[i])
            + 1e-10
        )
        similarity_scores.append(sim_score)

    plt.figure(figsize=(10, 8))
    plt.plot(similarity_scores, marker="o", linestyle="-", color="b")
    plt.title("Sliding Window Sentence Similarity Scores")
    plt.xlabel("Sentence Index")
    plt.ylabel("Similarity Score")
    plt.grid(True)
    plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    plt.show()

    # Each dip below the threshold is where a chunker would consider cutting.
    for i, score in enumerate(similarity_scores):
        if score < threshold:
            print(
                f"First sentence after similarity score "
                f"below {threshold}: {sentences[i + window_size]}"
            )
