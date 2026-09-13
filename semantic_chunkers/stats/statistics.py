from dataclasses import dataclass
from typing import List

from semantic_chunkers.schema import Chunk


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

    Why a chunk was cut is already recorded on the chunk, so the counts are
    read back off the result rather than tallied inside the chunking loop. A
    chunk cut by the similarity threshold carries ``is_triggered``; the others
    were cut by the token limit, except a trailing one, which is whatever was
    left over when the documents ran out.

    :param docs: The documents that were chunked.
    :param chunks: The chunks they produced, in order.

    :return: The statistics for this batch.
    """
    total_chunks = len(chunks)
    chunks_by_threshold = sum(1 for chunk in chunks if chunk.is_triggered)
    chunks_by_last_split = 1 if chunks and not chunks[-1].is_triggered else 0
    token_counts = [
        chunk.token_count for chunk in chunks if chunk.token_count is not None
    ]
    return ChunkStatistics(
        total_documents=len(docs),
        total_chunks=total_chunks,
        chunks_by_threshold=chunks_by_threshold,
        chunks_by_max_chunk_size=total_chunks
        - chunks_by_threshold
        - chunks_by_last_split,
        chunks_by_last_split=chunks_by_last_split,
        min_token_size=min(token_counts, default=0),
        max_token_size=max(token_counts, default=0),
        chunks_by_similarity_ratio=(
            chunks_by_threshold / total_chunks if total_chunks else 0
        ),
    )
