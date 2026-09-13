from typing import Any, List, Optional

from colorama import Fore, Style
from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator

from semantic_chunkers.encoders import DenseEncoder, EncoderError
from semantic_chunkers.schema import Chunk
from semantic_chunkers.splitters.base import BaseSplitter


class BaseChunker(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str
    # Validation is skipped because the annotation is a Protocol: an encoder
    # with no async path is still usable on the synchronous call, so requiring
    # the whole protocol here would reject it before it chunked anything.
    encoder: Optional[SkipValidation[DenseEncoder]] = None
    splitter: BaseSplitter

    @field_validator("encoder")
    @classmethod
    def check_encoder_is_callable(cls, v: Any) -> Any:
        if v is not None and not callable(v):
            raise EncoderError(
                f"An encoder must be callable as encoder(docs), and "
                f"{type(v).__name__} is not. Wrap a function in CallableEncoder."
            )
        return v

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
                f"Split {i + 1}, tokens {split.token_count}, triggered by: {triggered}"
            )
            print(colored_content)
            print("-" * 88)
            print("\n")
