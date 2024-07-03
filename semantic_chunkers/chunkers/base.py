from typing import Any, List

from colorama import Fore, Style
from pydantic.v1 import BaseModel, Extra
from semantic_router.encoders.base import BaseEncoder

from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters.base import BaseSplitter


class BaseChunker(BaseModel):
    name: str
    encoder: BaseEncoder
    splitter: BaseSplitter

    class Config:
        extra = Extra.allow

    def __call__(self, docs: List[str]) -> List[List[Chunk]]:
        raise NotImplementedError("Subclasses must implement this method")

    def _split(self, doc: str) -> List[str]:
        return self.splitter(doc)

    def _chunk(self, splits: List[Any]) -> List[Chunk]:
        raise NotImplementedError("Subclasses must implement this method")

    def print(self, document_splits: List[Chunk]) -> None:
        colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA]
        for i, split in enumerate(document_splits):
            color = colors[i % len(colors)]
            colored_content = f"{color}{split.content}{Style.RESET_ALL}"
            if split.is_triggered:
                triggered = f"{split.triggered_score:.2f}"
            elif i == len(document_splits) - 1:
                triggered = "final split"
            else:
                triggered = "token limit"
            print(
                f"Split {i + 1}, "
                f"tokens {split.token_count}, "
                f"triggered by: {triggered}"
            )
            print(colored_content)
            print("-" * 88)
            print("\n")
