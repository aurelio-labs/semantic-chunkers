from typing import Any, List, Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    """A run of consecutive splits, and the text of the document it covers.

    ``content`` is the document verbatim between ``start`` and ``end``,
    whitespace and all, so the chunks of one document join back into that
    document and a chunk can be highlighted in its source:

    ```python
    chunks = RegexChunker()(["First one.\\n\\n  Second one."])[0]
    "".join(chunk.content for chunk in chunks)   # the document, exactly
    doc[chunks[0].start : chunks[0].end] == chunks[0].content   # True
    ```

    ``splits`` holds the units the splitter produced, stripped, which is what
    the chunker compared when it decided where to cut.

    The three are ``None`` together for a chunk the library could not place in
    a source document: one built by hand, or one whose splits are not text —
    ``ConsecutiveChunker`` accepts a pre-split list of video frames, and those
    chunks carry the frames in ``splits`` and nothing in ``content``.
    """

    splits: List[Any]
    content: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None
