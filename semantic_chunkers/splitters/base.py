from typing import List, Tuple

from pydantic import BaseModel, ConfigDict


class BaseSplitter(BaseModel):
    model_config = ConfigDict(extra="allow")

    def __call__(self, doc: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

    def spans(self, doc: str) -> List[Tuple[int, int]]:
        """Locate every split of ``doc`` in the document.

        Returns one ``(start, end)`` pair per split, in the order
        ``__call__`` returned them, such that ``doc[start:end]`` is that
        split. Chunkers use the spans to give each chunk the exact text it
        covers, so a splitter that can be located gets `Chunk.content` and
        offsets for free.

        This default walks the document looking for each split in turn, which
        is correct for any splitter whose splits are verbatim substrings of
        the document in document order. A splitter that rewrites the text —
        lower-casing, normalising quotes — cannot be located that way and
        returns an empty list here, either by inheriting this behaviour or by
        overriding. Its chunks then carry no ``content``, ``start`` or
        ``end``, and everything else about them is unchanged.

        ```python
        RegexSplitter().spans("First line. Second line.")
        # [(0, 11), (12, 24)]
        ```
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for split in self(doc):
            start = doc.find(split, cursor)
            if start == -1:
                return []
            spans.append((start, start + len(split)))
            cursor = start + len(split)
        return spans
