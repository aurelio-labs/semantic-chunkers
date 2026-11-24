from typing import List, Tuple

from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.splitters.regex import _strip_span


class SentencexSplitter(BaseSplitter):
    """Split text into sentences with sentencex, behind the ``sentencex`` extra.

    sentencex is a rule-based segmenter from Wikimedia with abbreviation lists
    for many languages. Unlike `RegexSplitter`, it splits sentences that start
    with a non-ASCII capital or with no capital at all, as in Russian or
    Japanese, and it keeps a colon or an abbreviation such as ``"Dr."`` inside
    its sentence. When unsure it leaves two sentences joined rather than risk a
    wrong cut.

    ```bash
    pip install "semantic-chunkers[sentencex]"
    ```

    ```python
    SentencexSplitter(language="fr")("Il a quitté Paris. Élise est restée.")
    # ['Il a quitté Paris.', 'Élise est restée.']
    ```

    :param language: the language of the text as a lower-case, Wikipedia-style
        code such as ``"en"``, ``"de"``, ``"pt-br"`` or ``"zh-hans"``. A code
        sentencex has no rules for follows its fallback chain to the closest
        language it has, and ends at English, without an error.
    """

    language: str = "en"

    def __call__(self, doc: str) -> List[str]:
        """Split ``doc`` into sentences, each a verbatim, stripped substring of it.

        ```python
        SentencexSplitter()("Note: This only applies to version 2. Then stop.")
        # ['Note: This only applies to version 2.', 'Then stop.']
        ```
        """
        return [piece.strip() for piece in self._segment(doc) if piece.strip()]

    def spans(self, doc: str) -> List[Tuple[int, int]]:
        """Locate each sentence of ``doc``, as ``(start, end)`` offsets into it.

        The offsets count characters, not bytes, and are trimmed of surrounding
        whitespace, so ``[doc[s:e] for s, e in splitter.spans(doc)] ==
        splitter(doc)``. A sentence that is only whitespace is dropped.

        ```python
        SentencexSplitter().spans("First line.\\n\\nSecond line.")
        # [(0, 11), (13, 25)]
        ```
        """
        pieces = self._segment(doc)
        if "".join(pieces) != doc:
            # sentencex returns slices of the text, so this should not happen;
            # if it does, find each sentence in the text instead of guessing.
            return super().spans(doc)
        spans = []
        cursor = 0
        for piece in pieces:
            span = _strip_span(doc, cursor, cursor + len(piece))
            if span[0] < span[1]:
                spans.append(span)
            cursor += len(piece)
        return spans

    def _segment(self, doc: str) -> List[str]:
        """The pieces sentencex cuts ``doc`` into, whitespace included."""
        try:
            import sentencex
        except ImportError as error:
            raise ImportError(
                "SentencexSplitter needs sentencex: "
                'pip install "semantic-chunkers[sentencex]"'
            ) from error
        return list(sentencex.segment(self.language, doc))
