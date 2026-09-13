import asyncio
from typing import List, Optional, Union

import regex

from semantic_chunkers.chunkers.base import BaseChunker
from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters import RegexSplitter
from semantic_chunkers.utils import text


class RegexChunker(BaseChunker):
    def __init__(
        self,
        splitter: RegexSplitter = RegexSplitter(),
        max_chunk_tokens: int = 300,
        delimiters: Optional[List[Union[str, regex.Pattern]]] = None,
    ):
        super().__init__(name="regex_chunker", encoder=None, splitter=splitter)
        self.splitter: RegexSplitter = splitter
        self.max_chunk_tokens = max_chunk_tokens
        self.delimiters: List[Union[str, regex.Pattern]] = delimiters or []

    def __call__(self, docs: list[str]) -> List[List[Chunk]]:
        """Chunk each document separately, returning one list of chunks per document.

        ```python
        chunks_by_doc = RegexChunker()(["First document.", "Second document."])
        len(chunks_by_doc)  # 2
        ```
        """
        docs_chunks = []

        for doc in docs:
            chunks: List[Chunk] = []
            current_chunk = Chunk(splits=[], metadata={})
            current_chunk.token_count = 0

            sentences = self.splitter(doc, delimiters=self.delimiters)
            for sentence in sentences:
                sentence_token_count = text.tiktoken_length(sentence)
                if current_chunk.token_count is None:
                    raise ValueError("current_chunk.token_count is None, expected int")
                if (
                    current_chunk.token_count + sentence_token_count
                    > self.max_chunk_tokens
                ):
                    if current_chunk.splits:
                        chunks.append(current_chunk)
                    current_chunk = Chunk(splits=[])
                    current_chunk.token_count = 0

                current_chunk.splits.append(sentence)
                if current_chunk.token_count is None:
                    current_chunk.token_count = 0
                current_chunk.token_count += sentence_token_count

            # Last chunk of this document.
            if current_chunk.splits:
                chunks.append(current_chunk)

            docs_chunks.append(chunks)

        return docs_chunks

    async def acall(self, docs: list[str]) -> List[List[Chunk]]:
        chunks = await asyncio.to_thread(self.__call__, docs)
        return chunks
