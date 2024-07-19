import unittest

from semantic_chunkers.splitters.regex import RegexSplitter


class TestRegexSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = RegexSplitter()

    def test_split_by_double_newline(self):
        doc = "This is the first paragraph.\n\nThis is the second paragraph."
        expected = ["This is the first paragraph.", "This is the second paragraph."]
        result = self.splitter(doc, delimiters=["\n\n"])
        self.assertEqual(result, expected)

    def test_split_by_single_newline(self):
        doc = "This is the first line.\nThis is the second line."
        expected = ["This is the first line.", "This is the second line."]
        result = self.splitter(doc, delimiters=["\n"])
        self.assertEqual(result, expected)

    def test_split_by_period(self):
        doc = "This is the first sentence. This is the second sentence."
        expected = ["This is the first sentence.", "This is the second sentence."]
        result = self.splitter(doc, delimiters=["."])
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
        result = self.splitter(doc, delimiters=["\n\n", "\n", "."])
        self.assertEqual(result, expected)

    def test_custom_delimiters(self):
        doc = "First part|Second part|Third part"
        expected = ["First part|", "Second part|", "Third part"]
        result = self.splitter(doc, delimiters=["|"])
        self.assertEqual(result, expected)

    def test_regex_split(self):
        doc = "This is a sentence. And another one! Yet another?"
        expected = ["This is a sentence.", "And another one!", "Yet another?"]
        result = self.splitter(doc)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
