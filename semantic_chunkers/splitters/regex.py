from typing import List, Union

import regex

from semantic_chunkers.splitters.base import BaseSplitter


class RegexSplitter(BaseSplitter):
    """
    Enhanced regex pattern to split a given text into sentences more accurately.
    """

    regex_pattern = r"""
        # Negative lookbehind for word boundary, word char, dot, word char
        (?<!\b\w\.\w.)
        # Negative lookbehind for single uppercase initials like "A."
        (?<!\b[A-Z][a-z]\.)
        # Negative lookbehind for abbreviations like "U.S."
        (?<!\b[A-Z]\.)
        # Negative lookbehind for abbreviations with uppercase letters and dots
        (?<!\b\p{Lu}\.\p{Lu}.)
        # Negative lookbehind for numbers, to avoid splitting decimals
        (?<!\b\p{N}\.)
        # Positive lookbehind for punctuation followed by whitespace
        (?<=\.|\?|!|:|\.\.\.)\s+
        # Positive lookahead for uppercase letter or opening quote at word boundary
        (?="?(?=[A-Z])|"\b)
        # OR
        |
        # Splits after punctuation that follows closing punctuation, followed by
        # whitespace
        (?<=[\"\'\]\)\}][\.!?])\s+(?=[\"\'\(A-Z])
        # OR
        |
        # Splits after punctuation if not preceded by a period
        (?<=[^\.][\.!?])\s+(?=[A-Z])
        # OR
        |
        # Handles splitting after ellipses
        (?<=\.\.\.)\s+(?=[A-Z])
        # OR
        |
        # Matches and removes control characters and format characters
        [\p{Cc}\p{Cf}]+
        # OR
        |
        # Splits after punctuation marks followed by another punctuation mark
        (?<=[\.!?])(?=[\.!?])
        # OR
        |
        # Splits after exclamation or question marks followed by whitespace or end of string
        (?<=[!?])(?=\s|$)
    """

    def __call__(
        self, doc: str, delimiters: List[Union[str, regex.Pattern]] = []
    ) -> List[str]:
        if not delimiters:
            compiled_pattern = regex.compile(self.regex_pattern)
            delimiters.append(compiled_pattern)
        sentences = [doc]
        for delimiter in delimiters:
            sentences_for_next_delimiter = []
            for sentence in sentences:
                if isinstance(delimiter, regex.Pattern):
                    sub_sentences = regex.split(
                        self.regex_pattern, doc, flags=regex.VERBOSE
                    )
                    split_char = ""  # No single character to append for regex pattern
                else:
                    sub_sentences = sentence.split(delimiter)
                    split_char = delimiter
                for i, sub_sentence in enumerate(sub_sentences):
                    if i < len(sub_sentences) - 1:
                        sub_sentence += split_char  # Append delimiter to sub_sentence
                    if sub_sentence.strip():
                        sentences_for_next_delimiter.append(sub_sentence.strip())
            sentences = sentences_for_next_delimiter
        return sentences
