from typing import Any, List

import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_chunkers.schema import ChunkSet
from semantic_chunkers.chunkers.base import BaseChunker


class ConsecutiveChunker(BaseChunker):
    """
    Called "consecutive sim chunker" because we check the similarities of consecutive document embeddings (compare ith to i+1th document embedding).
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        name: str = "consecutive_chunker",
        score_threshold: float = 0.45,
    ):
        super().__init__(name=name, encoder=encoder)
        encoder.score_threshold = score_threshold
        self.score_threshold = score_threshold

    def __call__(self, docs: List[Any]) -> List[ChunkSet]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be split, if only wanted to
            split a single document, pass it as a list with a single element.

        :return: list of ChunkSet objects containing the chunks.
        """
        # Check if there's only a single document
        if len(docs) == 1:
            raise ValueError(
                "There is only one document provided; at least two are required to determine topics based on similarity."
            )

        doc_embeds = self.encoder(docs)
        norm_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeds, norm_embeds.T)
        total_docs = len(docs)
        splits = []
        curr_split_start_idx = 0
        curr_split_num = 1

        for idx in range(1, total_docs):
            curr_sim_score = sim_matrix[idx - 1][idx]
            if idx < len(sim_matrix) and curr_sim_score < self.score_threshold:
                splits.append(
                    ChunkSet(
                        docs=list(docs[curr_split_start_idx:idx]),
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_split_start_idx = idx
                curr_split_num += 1
        splits.append(ChunkSet(docs=list(docs[curr_split_start_idx:])))
        return splits
