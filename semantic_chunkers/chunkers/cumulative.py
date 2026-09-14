import asyncio
from typing import Any, Generator, List

import numpy as np
from pydantic import SkipValidation
from tqdm.auto import tqdm

from semantic_chunkers.chunkers.base import BaseChunker
from semantic_chunkers.encoders import DenseEncoder, acall_encoder
from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.splitters.regex import RegexSplitter


class CumulativeChunker(BaseChunker):
    """
    Called "cumulative sim" because we check the similarities of the
    embeddings of cumulative concatenated documents with the next document.
    """

    encoder: SkipValidation[DenseEncoder]

    def __init__(
        self,
        encoder: DenseEncoder,
        splitter: BaseSplitter = RegexSplitter(),
        name: str = "cumulative_chunker",
        score_threshold: float = 0.45,
    ):
        super().__init__(name=name, encoder=encoder, splitter=splitter)
        self.score_threshold = score_threshold

    @staticmethod
    def _batch_starts(splits: List[Any], batch_size: int) -> range:
        """Where each batch of splits to encode begins.

        Empty for a document of one split: that split has nothing to compare
        against, so `_cut` reads no embedding and encoding it would be paying
        for one nothing uses.
        """
        return range(0, len(splits), batch_size) if len(splits) > 1 else range(0)

    def _cut(
        self, splits: List[Any], split_embeds: np.ndarray
    ) -> Generator[str, Any, List[Chunk]]:
        """Walk the splits and decide where each chunk ends.

        The sync and the async path both drive this one copy of the decision,
        so where the chunks are cut cannot drift between them. It is a
        generator because the only step that differs between the two is the
        encoding: it yields the text of the running chunk, is sent back that
        text's embedding, and carries on. `splits` is already encoded in
        `split_embeds`, so a running chunk one split long is never yielded.

        :param splits: splits to be merged into chunks.
        :param split_embeds: one embedding per split, in order.

        :return: list of chunks, as the generator's return value.
        """
        chunks = []
        curr_chunk_start_idx = 0
        num_splits = len(splits)

        # Stop one short: the last split has no next document to compare with.
        for idx in tqdm(range(num_splits - 1)):
            if curr_chunk_start_idx == idx:
                # A running chunk of one split is that split, already encoded.
                curr_chunk_docs_embed = split_embeds[idx]
            else:
                # Otherwise compare the documents from the start of the chunk
                # up to the current one with the next.
                curr_chunk_docs_embed = yield "\n".join(
                    splits[curr_chunk_start_idx : idx + 1]
                )
            next_doc_embed = split_embeds[idx + 1]

            curr_sim_score = np.dot(curr_chunk_docs_embed, next_doc_embed) / (
                np.linalg.norm(curr_chunk_docs_embed) * np.linalg.norm(next_doc_embed)
            )
            # Decision to chunk based on similarity score.
            if curr_sim_score < self.score_threshold:
                chunks.append(
                    Chunk(
                        splits=list(splits[curr_chunk_start_idx : idx + 1]),
                        is_triggered=True,
                        triggered_score=curr_sim_score,
                    )
                )
                curr_chunk_start_idx = idx + 1  # Start index for the next segment.

        # Add the last segment after the loop.
        if curr_chunk_start_idx < num_splits:
            chunks.append(Chunk(splits=list(splits[curr_chunk_start_idx:])))

        return chunks

    def _chunk(self, splits: List[Any], batch_size: int = 64) -> List[Chunk]:
        """Merge splits into chunks using semantic similarity.

        :param splits: splits to be merged into chunks.
        :param batch_size: number of splits to encode in one request.

        :return: list of chunks.
        """
        split_embeds = np.array(
            [
                embed
                for i in self._batch_starts(splits, batch_size)
                for embed in self.encoder(splits[i : i + batch_size])
            ]
        )

        cuts = self._cut(splits, split_embeds)
        embed: Any = None  # nothing to send until the generator asks for a text
        while True:
            try:
                curr_chunk_docs = cuts.send(embed)
            except StopIteration as done:
                return done.value
            embed = self.encoder([curr_chunk_docs])[0]

    async def _async_chunk(
        self, splits: List[Any], batch_size: int = 64
    ) -> List[Chunk]:
        """Merge splits into chunks using semantic similarity.

        Cuts where `_chunk` cuts, and pays what `_chunk` pays: the splits are
        encoded once, in the same batches, only concurrently rather than one
        batch after another.

        :param splits: splits to be merged into chunks.
        :param batch_size: number of splits to encode in one request.

        :return: list of chunks.
        """
        batches = await asyncio.gather(
            *[
                acall_encoder(self.encoder, splits[i : i + batch_size])
                for i in self._batch_starts(splits, batch_size)
            ]
        )
        split_embeds = np.array([embed for batch in batches for embed in batch])

        cuts = self._cut(splits, split_embeds)
        embed: Any = None  # nothing to send until the generator asks for a text
        while True:
            try:
                curr_chunk_docs = cuts.send(embed)
            except StopIteration as done:
                return done.value
            embed = (await acall_encoder(self.encoder, [curr_chunk_docs]))[0]

    def __call__(self, docs: List[str]) -> List[List[Chunk]]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be chunk, if only wanted to
            chunk a single document, pass it as a list with a single element.

        :return: list of list objects containing the chunks.
        """
        all_chunks = []
        for doc in docs:
            # split the document into sentences (if needed), keeping track of
            # where each split sits in it
            splits, spans = self._split_spans(doc)
            doc_chunks = self._chunk(splits)
            all_chunks.append(self._attach_spans(doc, spans, doc_chunks))
        return all_chunks

    async def acall(self, docs: List[str]) -> List[List[Chunk]]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be chunk, if only wanted to
            chunk a single document, pass it as a list with a single element.

        :return: list of list objects containing the chunks.
        """
        all_chunks = []
        for doc in docs:
            # split the document into sentences (if needed), keeping track of
            # where each split sits in it
            splits, spans = self._split_spans(doc)
            doc_chunks = await self._async_chunk(splits)
            all_chunks.append(self._attach_spans(doc, spans, doc_chunks))
        return all_chunks
