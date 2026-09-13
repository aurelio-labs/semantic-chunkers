"""Statistics and plotting for the chunkers.

Nothing here decides where a document is cut. The chunkers delegate to it so
the algorithm modules stay free of chart formatting, and so matplotlib stays
out of the core install: it lives in the ``stats`` extra and is imported only
when a plot is actually drawn.

    pip install semantic-chunkers[stats]
"""

from semantic_chunkers.stats.plot import (
    plot_sentence_similarity_scores,
    plot_similarity_scores,
)
from semantic_chunkers.stats.statistics import ChunkStatistics, chunk_statistics

__all__ = [
    "ChunkStatistics",
    "chunk_statistics",
    "plot_sentence_similarity_scores",
    "plot_similarity_scores",
]
