import asyncio
from typing import List, Union

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
        delimiters: List[Union[str, regex.Pattern]] = [],
    ):
        super().__init__(name="regex_chunker", encoder=None, splitter=splitter)
        self.splitter: RegexSplitter = splitter
        self.max_chunk_tokens = max_chunk_tokens
        self.delimiters = delimiters

    def __call__(self, docs: list[str]) -> List[List[Chunk]]:
        chunks = []
        current_chunk = Chunk(
            splits=[],
            metadata={},
        )
        current_chunk.token_count = 0

        for doc in docs:
            sentences = self.splitter(doc, delimiters=self.delimiters)
            for sentence in sentences:
                sentence_token_count = text.tiktoken_length(sentence)
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

        # Last chunk
        if current_chunk.splits:
            chunks.append(current_chunk)

        return [chunks]

    async def acall(self, docs: list[str]) -> List[List[Chunk]]:
        chunks = await asyncio.to_thread(self.__call__, docs)
        return chunks
