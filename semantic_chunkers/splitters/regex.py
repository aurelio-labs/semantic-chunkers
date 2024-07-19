from typing import List

import regex

from semantic_chunkers.splitters.base import BaseSplitter


class RegexSplitter(BaseSplitter):
    """
    Enhanced regex pattern to split a given text into sentences more accurately.

    The enhanced regex pattern includes handling for:
    - Direct speech and quotations.
    - Abbreviations, initials, and acronyms.
    - Decimal numbers and dates.
    - Ellipses and other punctuation marks used in informal text.
    - Removing control characters and format characters.
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
    """

    def __call__(self, doc: str) -> List[str]:
        # Step 1: Split by \n\n
        chunks = doc.split("\n\n")
        sentences = []
        for chunk in chunks:
            # Step 2: Split by \n within each chunk
            sub_chunks = chunk.split("\n")
            for sub_chunk in sub_chunks:
                # Step 3: Split by regex pattern within each sub_chunk
                sub_sentences = regex.split(
                    self.regex_pattern, sub_chunk, flags=regex.VERBOSE
                )
                for sentence in sub_sentences:
                    if sentence.strip():
                        sentences.append(sentence.strip())
        return sentences
