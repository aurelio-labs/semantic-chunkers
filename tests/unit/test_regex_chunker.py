import asyncio
import unittest

from semantic_chunkers.chunkers.regex import RegexChunker
from semantic_chunkers.schema import Chunk
from semantic_chunkers.utils import text


class TestRegexChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = RegexChunker(max_chunk_tokens=10)

    def test_call(self):
        docs = ["This is a test. This is only a test."]
        chunks_list = self.chunker(docs)
        chunks = chunks_list[0]

        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
        self.assertGreater(len(chunks), 0)
        self.assertTrue(
            all(
                text.tiktoken_length(chunk.content) <= self.chunker.max_chunk_tokens
                for chunk in chunks
            )
        )

    def test_acall(self):
        docs = ["This is a test. This is only a test."]

        async def run_test():
            chunks_list = await self.chunker.acall(docs)
            chunks = chunks_list[0]
            self.assertIsInstance(chunks, list)
            self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))
            self.assertGreater(len(chunks), 0)
            self.assertTrue(
                all(
                    text.tiktoken_length(chunk.content) <= self.chunker.max_chunk_tokens
                    for chunk in chunks
                )
            )

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
