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

    def test_regex_chunker_keeps_documents_separate(self):
        docs = ["Alpha one. Alpha two.", "Beta one. Beta two."]
        chunks_list = RegexChunker(max_chunk_tokens=300)(docs)

        self.assertEqual(len(chunks_list), 2)
        first, second = (
            [split for chunk in chunks for split in chunk.splits]
            for chunks in chunks_list
        )
        self.assertEqual(first, ["Alpha one.", "Alpha two."])
        self.assertEqual(second, ["Beta one.", "Beta two."])

        # No single chunk may mix sentences from two source documents.
        for chunks, own, other in ((chunks_list[0], "Alpha", "Beta"), (chunks_list[1], "Beta", "Alpha")):
            for chunk in chunks:
                self.assertIn(own, chunk.content)
                self.assertNotIn(other, chunk.content)

    def test_regex_chunker_matches_other_chunkers_return_shape(self):
        for docs in ([], ["One."], ["One.", "Two.", "Three."]):
            with self.subTest(n=len(docs)):
                self.assertEqual(len(self.chunker(docs)), len(docs))

    def test_regex_chunker_acall_matches_call(self):
        docs = ["Alpha one. Alpha two.", "Beta one. Beta two."]
        sync_result = self.chunker(docs)
        async_result = asyncio.run(self.chunker.acall(docs))

        self.assertEqual(
            [[chunk.splits for chunk in chunks] for chunks in sync_result],
            [[chunk.splits for chunk in chunks] for chunks in async_result],
        )


if __name__ == "__main__":
    unittest.main()
