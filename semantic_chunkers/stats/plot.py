"""Charts for inspecting how a chunker decided where to cut.

matplotlib is not a core dependency, so every function here imports it through
:func:`_pyplot` as its first act: no encoder call is spent before a missing
install is reported.
"""

from typing import Callable, List

import numpy as np

from semantic_chunkers.schema import Chunk


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
    """Plot the similarity score of each split and the size of each chunk.

    The upper axes show the rolling-window similarity with the chosen
    threshold and every cut marked; the lower axes show the token count of each
    resulting chunk.

    :param similarities: Similarity score of each split against its window.
    :param split_indices: Indices the chunker decided to cut after.
    :param chunks: The chunks those cuts produced.
    :param calculated_threshold: Threshold the scores were compared against.
    :param window_size: Number of preceding splits in the rolling window.

    :raises ImportError: When matplotlib is not installed.
    """
    plt = _pyplot()

    _, axs = plt.subplots(2, 1, figsize=(12, 12))  # Adjust for two plots

    # Plot 1: Similarity Scores
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

    # Annotating each similarity score
    for i, score in enumerate(similarities):
        axs[0].annotate(
            f"{score:.2f}",  # Formatting to two decimal places
            (i, score),
            textcoords="offset points",
            xytext=(0, 10),  # Positioning the text above the point
            ha="center",
        )  # Center-align the text

    axs[0].set_xlabel("Document Segment Index")
    axs[0].set_ylabel("Similarity Score")
    axs[0].set_title(
        f"Threshold: {calculated_threshold} | Window Size: {window_size}",
        loc="right",
        fontsize=10,
    )
    axs[0].legend()

    # Plot 2: Chunk Token Size Distribution
    token_counts = [split.token_count for split in chunks]
    axs[1].bar(range(len(token_counts)), token_counts, color="lightblue")
    axs[1].set_title("Chunk Token Sizes")
    axs[1].set_xlabel("Chunk Index")
    axs[1].set_ylabel("Token Count")
    axs[1].set_xticks(range(len(token_counts)))
    axs[1].set_xticklabels([str(i) for i in range(len(token_counts))])
    axs[1].grid(True)

    # Annotate each bar with the token size
    for idx, token_count in enumerate(token_counts):
        if not token_count:
            continue
        axs[1].text(idx, token_count + 0.01, str(token_count), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def plot_sentence_similarity_scores(
    sentences: List[str],
    encode: Callable[[List[str]], np.ndarray],
    threshold: float,
    window_size: int,
) -> None:
    """Plot how each sentence compares with the ``window_size`` before it.

    Computes similarity scores between the average of the last ``window_size``
    sentences and the next one, plots a graph of these similarity scores, and
    prints the first sentence after a similarity score below ``threshold``.

    :param sentences: The sentences to compare, in document order.
    :param encode: Callable turning those sentences into embeddings. It is
        called only once matplotlib is known to be importable.
    :param threshold: Score below which a sentence is reported as a boundary.
    :param window_size: Number of preceding sentences averaged into the window.

    :raises ImportError: When matplotlib is not installed.
    """
    plt = _pyplot()

    encoded_sentences = encode(sentences)
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

    for i, score in enumerate(similarity_scores):
        if score < threshold:
            print(
                f"First sentence after similarity score "
                f"below {threshold}: {sentences[i + window_size]}"
            )
