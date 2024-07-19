import unittest

from semantic_chunkers.splitters.regex import RegexSplitter


class TestRegexSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = RegexSplitter()

    def test_split_by_double_newline(self):
        doc = "This is the first paragraph.\n\nThis is the second paragraph."
        expected = ["This is the first paragraph.", "This is the second paragraph."]
        result = self.splitter(doc)
        self.assertEqual(result, expected)

    def test_split_by_single_newline(self):
        doc = "This is the first line.\nThis is the second line."
        expected = ["This is the first line.", "This is the second line."]
        result = self.splitter(doc)
        self.assertEqual(result, expected)

    def test_split_by_period(self):
        doc = "This is the first sentence. This is the second sentence."
        expected = ["This is the first sentence.", "This is the second sentence."]
        result = self.splitter(doc)
        self.assertEqual(result, expected)

    def test_complex_split(self):
        doc = """
        First paragraph.\n\nSecond paragraph.\nThird line in second paragraph. Fourth line.\n\nFifth paragraph."""
        expected = [
            "First paragraph.",
            "Second paragraph.",
            "Third line in second paragraph.",
            "Fourth line.",
            "Fifth paragraph.",
        ]
        result = self.splitter(doc)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
