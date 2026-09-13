from typing import Any, List, Optional, Tuple

from colorama import Fore, Style
from pydantic import BaseModel, ConfigDict, Field, field_validator
from semantic_router.encoders.base import DenseEncoder

from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.utils.logger import logger


class BaseChunker(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str
    encoder: Optional[DenseEncoder] = Field(default=None, validate_default=True)
    splitter: BaseSplitter

    @field_validator("encoder", mode="before")
    @classmethod
    def set_encoder(cls, v):
        if v is None:
            return DenseEncoder(name="default")
        return v

    def __call__(self, docs: List[str]) -> List[List[Chunk]]:
        raise NotImplementedError("Subclasses must implement this method")

    def _split(self, doc: str) -> List[str]:
        return self.splitter(doc)

    def _split_spans(self, doc: Any) -> Tuple[List[Any], List[Tuple[int, int]]]:
        """Split ``doc`` into units, and say where in ``doc`` each one came from.

        The spans are what `Chunk.content` and the chunk offsets are built
        from. They come back empty when there are none to be had: a document
        that is not a ``str`` is already a list of units — the video notebook
        passes decoded frames — and a splitter that rewrites its text cannot
        be located in the document it was given. Chunking is the same either
        way; only the offsets are lost.
        """
        if not isinstance(doc, str):
            return doc, []
        spans = self.splitter.spans(doc)
        if not spans:
            # An empty document, or a splitter that cannot be located. Ask for
            # the splits themselves rather than assume the document has none.
            return self._split(doc), []
        return [doc[start:end] for start, end in spans], spans

    def _attach_spans(
        self, doc: Any, spans: List[Tuple[int, int]], chunks: List[Chunk]
    ) -> List[Chunk]:
        """Give each chunk its offsets into ``doc`` and the text it covers.

        A chunk runs from the start of its first split to the start of the
        next chunk's first split, so the whitespace between two splits belongs
        to the chunk on its left. The first chunk starts at 0 and the last ends
        at the end of the document. Every character therefore lands in exactly
        one chunk, and joining the chunks' ``content`` reproduces ``doc``.
        """
        if not isinstance(doc, str):
            return chunks
        if sum(len(chunk.splits) for chunk in chunks) != len(spans):
            # The chunker added or dropped splits, so which of them a chunk
            # holds is no longer a question of counting. Offsets that might be
            # wrong are worse than none at all.
            logger.warning(
                "Chunk splits do not line up with the document's splits; "
                "returning chunks without content or offsets."
            )
            return chunks
        start = 0
        cursor = 0
        for position, chunk in enumerate(chunks):
            cursor += len(chunk.splits)
            last = position == len(chunks) - 1
            end = len(doc) if last or cursor >= len(spans) else spans[cursor][0]
            chunk.start, chunk.end, chunk.content = start, end, doc[start:end]
            start = end
        return chunks

    def _chunk(self, splits: List[Any]) -> List[Chunk]:
        raise NotImplementedError("Subclasses must implement this method")

    def print(self, document_splits: List[Chunk]) -> None:
        colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA]
        for i, split in enumerate(document_splits):
            color = colors[i % len(colors)]
            colored_content = f"{color}{split.content or ''}{Style.RESET_ALL}"
            if split.is_triggered:
                triggered = f"{split.triggered_score:.2f}"
            elif i == len(document_splits) - 1:
                triggered = "final split"
            else:
                triggered = "token limit"
            print(
                f"Split {i + 1}, tokens {split.token_count}, triggered by: {triggered}"
            )
            print(colored_content)
            print("-" * 88)
            print("\n")
