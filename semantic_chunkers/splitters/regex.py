from typing import List, Optional, Tuple, Union

import regex

from semantic_chunkers.splitters.base import BaseSplitter


def _strip_span(doc: str, start: int, end: int) -> Tuple[int, int]:
    """The span of ``doc[start:end].strip()``, in document coordinates."""
    while start < end and doc[start].isspace():
        start += 1
    while end > start and doc[end - 1].isspace():
        end -= 1
    return start, end


class RegexSplitter(BaseSplitter):
    """
    Enhanced regex pattern to split a given text into sentences more accurately.
    """

    regex_pattern: str = r"""
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
        self,
        doc: str,
        delimiters: Optional[List[Union[str, regex.Pattern]]] = None,
    ) -> List[str]:
        """Split ``doc`` into sentences, applying each delimiter in turn.

        Each delimiter refines the output of the one before it, so the
        delimiters narrow the document progressively rather than each one
        re-splitting the whole document.

        A compiled delimiter is split with its own flags. ``regex_pattern`` is
        written in verbose form, so pass it as
        ``regex.compile(splitter.regex_pattern, flags=regex.VERBOSE)``; compiled
        without the flag it matches nothing and the document comes back whole.

        ```python
        RegexSplitter()("First line.\\nSecond line. Third line.", ["\\n"])
        # ['First line.', 'Second line. Third line.']
        ```
        """
        return [doc[start:end] for start, end in self.spans(doc, delimiters)]

    def spans(
        self,
        doc: str,
        delimiters: Optional[List[Union[str, regex.Pattern]]] = None,
    ) -> List[Tuple[int, int]]:
        """Locate each split of ``doc``, as ``(start, end)`` offsets into it.

        The same splits ``__call__`` returns, kept where they were found:
        ``[doc[s:e] for s, e in splitter.spans(doc)] == splitter(doc)``.

        The spans do not tile the document. What a split is stripped of, and
        what the delimiters matched, lies in the gaps between them — that is
        where paragraph breaks and indentation live. A caller that needs the
        document back whole, as `Chunk.content` does, runs each span up to
        the start of the next one.

        ```python
        RegexSplitter().spans("First line.\\n\\n  Second line.", ["\\n"])
        # [(0, 11), (15, 28)]
        ```
        """
        if not delimiters:
            delimiters = [regex.compile(self.regex_pattern, flags=regex.VERBOSE)]
        spans = [(0, len(doc))]
        for delimiter in delimiters:
            spans_for_next_delimiter = []
            for start, end in spans:
                for piece_start, piece_end in self._pieces(doc, start, end, delimiter):
                    span = _strip_span(doc, piece_start, piece_end)
                    if span[0] < span[1]:
                        spans_for_next_delimiter.append(span)
            spans = spans_for_next_delimiter
        return spans

    @staticmethod
    def _pieces(
        doc: str, start: int, end: int, delimiter: Union[str, regex.Pattern]
    ) -> List[Tuple[int, int]]:
        """Split ``doc[start:end]`` on one delimiter, as spans into ``doc``.

        A string delimiter stays on the end of the piece it follows and a
        compiled pattern is dropped, which is what the splitter has always
        done. The delimiter is applied to the slice rather than to the whole
        document so that a lookbehind sees only the piece it is splitting.
        """
        text = doc[start:end]
        cuts: List[Tuple[int, int]] = []  # (end of the piece, where the next begins)
        if isinstance(delimiter, regex.Pattern):
            cuts = [(match.start(), match.end()) for match in delimiter.finditer(text)]
        else:
            if not delimiter:
                raise ValueError("empty separator")
            index = text.find(delimiter)
            while index != -1:
                resume = index + len(delimiter)
                cuts.append((resume, resume))
                index = text.find(delimiter, resume)
        pieces = []
        cursor = 0
        for piece_end, resume in cuts:
            pieces.append((start + cursor, start + piece_end))
            cursor = resume
        pieces.append((start + cursor, end))
        return pieces
