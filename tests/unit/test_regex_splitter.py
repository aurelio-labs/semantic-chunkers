import inspect
import unittest

import regex

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

    def test_regex_splitter_applies_delimiters_in_sequence(self):
        doc = "Alpha one. Alpha two.\nBeta one. Beta two."
        sentence_pattern = regex.compile(self.splitter.regex_pattern, regex.VERBOSE)

        result = self.splitter(doc, delimiters=["\n", sentence_pattern])

        # Each piece appears exactly once: the second delimiter refines the
        # output of the first instead of re-splitting the whole document.
        self.assertEqual(
            result, ["Alpha one.", "Alpha two.", "Beta one.", "Beta two."]
        )
        self.assertEqual(len(result), len(set(result)))
        # The output covers the document exactly once, with no repeats.
        self.assertEqual(
            regex.sub(r"\s+", "", "".join(result)), regex.sub(r"\s+", "", doc)
        )

    def test_regex_splitter_honours_a_custom_compiled_pattern(self):
        result = self.splitter("a1b2c3", delimiters=[regex.compile(r"\d")])
        self.assertEqual(result, ["a", "b", "c"])

    def test_regex_splitter_default_delimiters_not_mutated(self):
        doc = "This is a sentence. And another one!"
        default = inspect.signature(RegexSplitter.__call__).parameters["delimiters"]

        before = default.default
        first = self.splitter(doc)
        second = self.splitter(doc)

        self.assertEqual(first, second)
        self.assertFalse(before)
        self.assertEqual(
            inspect.signature(RegexSplitter.__call__).parameters["delimiters"].default,
            before,
        )


if __name__ == "__main__":
    unittest.main()
