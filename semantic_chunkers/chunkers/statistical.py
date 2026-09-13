import asyncio
from typing import Any, List, Optional

import numpy as np
from pydantic import SkipValidation
from tqdm.auto import tqdm

from semantic_chunkers.chunkers.base import BaseChunker
from semantic_chunkers.encoders import DenseEncoder, acall_encoder
from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.splitters.regex import RegexSplitter
from semantic_chunkers.stats import ChunkStatistics, chunk_statistics
from semantic_chunkers.utils.logger import logger
from semantic_chunkers.utils.text import (
    async_retry_with_timeout,
    tiktoken_length,
    time_it,
)


class StatisticalChunker(BaseChunker):
    encoder: SkipValidation[DenseEncoder]

    def __init__(
        self,
        encoder: DenseEncoder,
        splitter: BaseSplitter = RegexSplitter(),
        name="statistical_chunker",
        threshold_adjustment=0.01,
        dynamic_threshold: bool = True,
        window_size=5,
        min_split_tokens=100,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        plot_chunks=False,
        enable_statistics=False,
    ):
        super().__init__(name=name, encoder=encoder, splitter=splitter)
        self.encoder = encoder
        self.threshold_adjustment = threshold_adjustment
        self.dynamic_threshold = dynamic_threshold
        self.window_size = window_size
        self.plot_chunks = plot_chunks
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.enable_statistics = enable_statistics
        self.statistics: ChunkStatistics
        self.DEFAULT_THRESHOLD = 0.5

    @time_it
    def _chunk(
        self, splits: List[Any], batch_size: int = 64, enforce_max_tokens: bool = False
    ) -> List[Chunk]:
        """Merge splits into chunks using semantic similarity, with optional enforcement
        of maximum token limits per chunk.

        :param splits: Splits to be merged into chunks.
        :param batch_size: Number of splits to process in one batch.
        :param enforce_max_tokens: If True, further split chunks that exceed the maximum
        token limit.

        :return: List of chunks.
        """
        if enforce_max_tokens:
            splits = self._split_oversized(splits)

        chunks: List[Chunk] = []
        last_chunk: Optional[Chunk] = None
        for i in tqdm(range(0, len(splits), batch_size)):
            batch_splits = splits[i : i + batch_size]
            if last_chunk is not None:
                batch_splits = last_chunk.splits + batch_splits

            doc_chunks = self._chunk_batch(
                batch_splits, self._encode_documents(batch_splits)
            )
            # The last chunk of a batch may still grow, so it leads the next
            # one rather than being finished here.
            chunks.extend(doc_chunks[:-1])
            last_chunk = doc_chunks[-1]

        if last_chunk:
            chunks.append(last_chunk)

        return chunks

    @time_it
    async def _async_chunk(
        self, splits: List[Any], batch_size: int = 64, enforce_max_tokens: bool = False
    ) -> List[Chunk]:
        """Merge splits into chunks using semantic similarity, with optional enforcement
        of maximum token limits per chunk.

        Produces the chunks `_chunk` produces, batching the same way: the chunk
        a batch leaves open leads the next batch, so a chunk can span a batch
        boundary. Only the encoding is concurrent, and every split is encoded
        once — a split carried into the next batch keeps the embedding it
        already has instead of being paid for twice.

        :param splits: Splits to be merged into chunks.
        :param batch_size: Number of splits to process in one batch.
        :param enforce_max_tokens: If True, further split chunks that exceed the maximum
        token limit.

        :return: List of chunks.
        """
        if enforce_max_tokens:
            splits = self._split_oversized(splits)

        if not splits:
            return []

        encoded_batches = await asyncio.gather(
            *[
                self._async_encode_documents(splits[i : i + batch_size])
                for i in range(0, len(splits), batch_size)
            ]
        )
        encoded_splits = np.concatenate(encoded_batches)

        chunks: List[Chunk] = []
        last_chunk: Optional[Chunk] = None
        for i in tqdm(range(0, len(splits), batch_size)):
            # The splits of the chunk carried over are the ones immediately
            # before `i`, so one slice picks out both the batch's splits and
            # their embeddings.
            carried = len(last_chunk.splits) if last_chunk is not None else 0
            batch_splits = splits[i - carried : i + batch_size]

            doc_chunks = self._chunk_batch(
                batch_splits, encoded_splits[i - carried : i + batch_size]
            )
            chunks.extend(doc_chunks[:-1])
            last_chunk = doc_chunks[-1]

        if last_chunk:
            chunks.append(last_chunk)

        return chunks

    def _split_oversized(self, splits: List[Any]) -> List[Any]:
        """Sentence-split every split that is already over the token limit."""
        new_splits = []
        for split in splits:
            if tiktoken_length(split) > self.max_split_tokens:
                logger.info(
                    f"Single document exceeds the maximum token limit "
                    f"of {self.max_split_tokens}. "
                    "Splitting to sentences before semantically merging."
                )
                new_splits.extend(self._split(split))
            else:
                new_splits.append(split)
        return [split for split in new_splits if split and split.strip()]

    def _chunk_batch(
        self, batch_splits: List[Any], encoded_splits: np.ndarray
    ) -> List[Chunk]:
        """Cut one batch of already-encoded splits into chunks.

        The sync and the async path both come through here, so where a batch
        is cut cannot drift between them.
        """
        similarities = self._calculate_similarity_scores(encoded_splits)

        if self.dynamic_threshold:
            calculated_threshold = self._find_optimal_threshold(
                batch_splits, similarities
            )
        else:
            calculated_threshold = self._static_threshold()
        split_indices = self._find_split_indices(
            similarities=similarities, calculated_threshold=calculated_threshold
        )

        doc_chunks = self._split_documents(
            docs=batch_splits,
            split_indices=split_indices,
            similarities=similarities,
        )

        if self.plot_chunks:
            self.plot_similarity_scores(
                similarities=similarities,
                split_indices=split_indices,
                chunks=doc_chunks,
                calculated_threshold=calculated_threshold,
            )

        if self.enable_statistics:
            print(self.statistics)

        return doc_chunks

    @time_it
    def __call__(self, docs: List[str], batch_size: int = 64) -> List[List[Chunk]]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be split, if only wanted to
            split a single document, pass it as a list with a single element.

        :return: list of Chunk objects containing the split documents.
        """
        if not docs:
            raise ValueError("At least one document is required for splitting.")

        all_chunks = []
        for doc in docs:
            token_count = tiktoken_length(doc)
            if token_count > self.max_split_tokens:
                logger.info(
                    f"Single document exceeds the maximum token limit "
                    f"of {self.max_split_tokens}. "
                    "Splitting to sentences before semantically merging."
                )
            if isinstance(doc, str):
                splits, spans = self._split_spans(doc)
                doc_chunks = self._chunk(splits, batch_size=batch_size)
                all_chunks.append(self._attach_spans(doc, spans, doc_chunks))
            else:
                raise ValueError("The document must be a string.")
        return all_chunks

    @time_it
    async def acall(self, docs: List[str], batch_size: int = 64) -> List[List[Chunk]]:
        """Split documents into smaller chunks based on semantic similarity.

        :param docs: list of text documents to be split, if only wanted to
            split a single document, pass it as a list with a single element.

        :return: list of Chunk objects containing the split documents.
        """
        if not docs:
            raise ValueError("At least one document is required for splitting.")

        all_chunks = []
        for doc in docs:
            token_count = tiktoken_length(doc)
            if token_count > self.max_split_tokens:
                logger.info(
                    f"Single document exceeds the maximum token limit "
                    f"of {self.max_split_tokens}. "
                    "Splitting to sentences before semantically merging."
                )
            if isinstance(doc, str):
                splits, spans = self._split_spans(doc)
                doc_chunks = await self._async_chunk(splits, batch_size=batch_size)
                all_chunks.append(self._attach_spans(doc, spans, doc_chunks))
            else:
                raise ValueError("The document must be a string.")
        return all_chunks

    def _static_threshold(self) -> float:
        """The threshold used when ``dynamic_threshold`` is off.

        The encoder protocol does not ask for a ``score_threshold``, so an
        encoder that carries one still decides the threshold and one that does
        not falls back to the default.
        """
        return getattr(self.encoder, "score_threshold", None) or self.DEFAULT_THRESHOLD

    @time_it
    def _encode_documents(self, docs: List[str]) -> np.ndarray:
        """
        Encodes a list of documents into embeddings. If the number of documents
        exceeds 2000, the documents are split into batches to avoid overloading
        the encoder. OpenAI has a limit of len(array) < 2048.

        :param docs: List of text documents to be encoded.
        :return: A numpy array of embeddings for the given documents.
        """
        max_docs_per_batch = 2000
        embeddings: List[Any] = []

        for i in range(0, len(docs), max_docs_per_batch):
            batch_docs = docs[i : i + max_docs_per_batch]
            try:
                batch_embeddings = self.encoder(batch_docs)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding documents {batch_docs}: {e}")
                raise

        return np.array(embeddings)

    # One attempt covers up to 2000 documents against a remote encoder, so the
    # budget is a minute rather than the five seconds it used to be. A stall
    # that outlasts all three attempts is raised, not returned as no embeddings.
    @async_retry_with_timeout(retries=3, timeout=60)
    @time_it
    async def _async_encode_documents(self, docs: List[str]) -> np.ndarray:
        """
        Encodes a list of documents into embeddings. If the number of documents
        exceeds 2000, the documents are split into batches to avoid overloading
        the encoder. OpenAI has a limit of len(array) < 2048.

        :param docs: List of text documents to be encoded.
        :return: A numpy array of embeddings for the given documents.
        """
        max_docs_per_batch = 2000
        embeddings: List[Any] = []

        for i in range(0, len(docs), max_docs_per_batch):
            batch_docs = docs[i : i + max_docs_per_batch]
            try:
                batch_embeddings = await acall_encoder(self.encoder, batch_docs)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding documents {batch_docs}: {e}")
                raise

        return np.array(embeddings)

    def _calculate_similarity_scores(self, encoded_docs: np.ndarray) -> List[float]:
        raw_similarities = []
        for idx in range(1, len(encoded_docs)):
            window_start = max(0, idx - self.window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                np.linalg.norm(cumulative_context) * np.linalg.norm(encoded_docs[idx])
                + 1e-10
            )
            raw_similarities.append(curr_sim_score)
        return raw_similarities

    def _find_split_indices(
        self, similarities: List[float], calculated_threshold: float
    ) -> List[int]:
        split_indices = []
        for idx, score in enumerate(similarities):
            logger.debug(f"Similarity score at index {idx}: {score}")
            if score < calculated_threshold:
                logger.debug(
                    f"Adding to split_indices due to score < threshold: "
                    f"{score} < {calculated_threshold}"
                )
                # Chunk after the document at idx
                split_indices.append(idx + 1)
        return split_indices

    def _find_optimal_threshold(self, docs: List[str], similarity_scores: List[float]):
        token_counts = [tiktoken_length(doc) for doc in docs]
        cumulative_token_counts = np.cumsum([0] + token_counts)

        # Analyze the distribution of similarity scores to set initial bounds
        median_score = np.median(similarity_scores)
        std_dev = np.std(similarity_scores)

        # Set initial bounds based on median and standard deviation
        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))

        iteration = 0
        median_tokens = 0
        calculated_threshold = 0.0
        while low <= high:
            calculated_threshold = (low + high) / 2
            split_indices = self._find_split_indices(
                similarity_scores, calculated_threshold
            )
            logger.debug(
                f"Iteration {iteration}: Trying threshold: {calculated_threshold}"
            )

            # Calculate the token counts for each split using the cumulative sums
            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip(
                    [0] + split_indices, split_indices + [len(token_counts)]
                )
            ]

            # Calculate the median token count for the chunks
            median_tokens = np.median(split_token_counts)
            logger.debug(
                f"Iteration {iteration}: Median tokens per split: {median_tokens}"
            )
            if (
                self.min_split_tokens - self.split_tokens_tolerance
                <= median_tokens
                <= self.max_split_tokens + self.split_tokens_tolerance
            ):
                logger.debug("Median tokens in target range. Stopping iteration.")
                break
            elif median_tokens < self.min_split_tokens:
                high = calculated_threshold - self.threshold_adjustment
                logger.debug(f"Iteration {iteration}: Adjusting high to {high}")
            else:
                low = calculated_threshold + self.threshold_adjustment
                logger.debug(f"Iteration {iteration}: Adjusting low to {low}")
            iteration += 1

        logger.debug(
            f"Optimal threshold {calculated_threshold} found "
            f"with median tokens ({median_tokens}) in target range "
            f"({self.min_split_tokens}-{self.max_split_tokens})."
        )

        return calculated_threshold

    def _split_documents(
        self, docs: List[str], split_indices: List[int], similarities: List[float]
    ) -> List[Chunk]:
        """
        This method iterates through each document, appending it to the current split
        until it either reaches a split point (determined by split_indices) or exceeds
        the maximum token limit for a split (self.max_split_tokens).
        When a document causes the current token count to exceed this limit,
        or when a split point is reached and the minimum token requirement is met,
        the current split is finalized and added to the List of chunks.
        """
        token_counts = [tiktoken_length(doc) for doc in docs]
        chunks, current_split = [], []
        current_tokens_count = 0

        for doc_idx, doc in enumerate(docs):
            doc_token_count = token_counts[doc_idx]
            logger.debug(f"Accumulative token count: {current_tokens_count} tokens")
            logger.debug(f"Document token count: {doc_token_count} tokens")
            # Check if current index is a split point based on similarity
            if doc_idx + 1 in split_indices:
                if (
                    self.min_split_tokens
                    <= current_tokens_count + doc_token_count
                    < self.max_split_tokens
                ):
                    # Include the current document before splitting
                    # if it doesn't exceed the max limit
                    current_split.append(doc)
                    current_tokens_count += doc_token_count

                    triggered_score = (
                        similarities[doc_idx] if doc_idx < len(similarities) else None
                    )
                    chunks.append(
                        Chunk(
                            splits=current_split.copy(),
                            is_triggered=True,
                            triggered_score=triggered_score,
                            token_count=current_tokens_count,
                        )
                    )
                    logger.debug(
                        f"Chunk finalized with {current_tokens_count} tokens due to "
                        f"threshold {triggered_score}."
                    )
                    current_split, current_tokens_count = [], 0
                    continue  # Move to the next document after splitting

            # Check if adding the current document exceeds the max token limit
            if current_tokens_count + doc_token_count > self.max_split_tokens:
                if current_tokens_count >= self.min_split_tokens:
                    chunks.append(
                        Chunk(
                            splits=current_split.copy(),
                            is_triggered=False,
                            triggered_score=None,
                            token_count=current_tokens_count,
                        )
                    )
                    logger.debug(
                        f"Chunk finalized with {current_tokens_count} tokens due to "
                        f"exceeding token limit of {self.max_split_tokens}."
                    )
                    current_split, current_tokens_count = [], 0

            current_split.append(doc)
            current_tokens_count += doc_token_count

        # Handle the last split
        if current_split:
            chunks.append(
                Chunk(
                    splits=current_split.copy(),
                    is_triggered=False,
                    triggered_score=None,
                    token_count=current_tokens_count,
                )
            )
            logger.debug(
                f"Final split added with {current_tokens_count} "
                "tokens due to remaining documents."
            )

        # Validation to ensure no tokens are lost during the split
        original_token_count = sum(token_counts)
        split_token_count = sum(
            [tiktoken_length(doc) for split in chunks for doc in split.splits]
        )
        if original_token_count != split_token_count:
            logger.error(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )
            raise ValueError(
                f"Token count mismatch: {original_token_count} != {split_token_count}"
            )

        # Counts are read back off the finished chunks, not tallied in the loop.
        self.statistics = chunk_statistics(docs, chunks)

        return chunks

    def plot_similarity_scores(
        self,
        similarities: List[float],
        split_indices: List[int],
        chunks: List[Chunk],
        calculated_threshold: float,
    ) -> None:
        """Plot the similarity scores and the chunk sizes of one batch.

        Drawn by `semantic_chunkers.stats`, which raises without the plotting
        extra: `pip install semantic-chunkers[stats]`.
        """
        from semantic_chunkers.stats import plot_similarity_scores

        plot_similarity_scores(
            similarities, split_indices, chunks, calculated_threshold, self.window_size
        )

    def plot_sentence_similarity_scores(
        self, docs: List[str], threshold: float, window_size: int
    ) -> None:
        """Plot how each sentence of ``docs`` compares with the ones before it.

        Drawn by `semantic_chunkers.stats`, which raises without the plotting
        extra: `pip install semantic-chunkers[stats]`. It encodes the sentences
        itself, and only once that extra is known to be present.
        """
        from semantic_chunkers.stats import plot_sentence_similarity_scores

        sentences = [sentence for doc in docs for sentence in self._split(doc)]
        plot_sentence_similarity_scores(
            sentences, self._encode_documents, threshold, window_size
        )
