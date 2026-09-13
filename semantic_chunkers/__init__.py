from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from semantic_chunkers.chunkers import (
    BaseChunker,
    ConsecutiveChunker,
    CumulativeChunker,
    RegexChunker,
    StatisticalChunker,
)
from semantic_chunkers.encoders import CallableEncoder, DenseEncoder, EncoderError
from semantic_chunkers.splitters import BaseSplitter, RegexSplitter

if TYPE_CHECKING:  # pragma: no cover - for type checkers, not for runtime
    from semantic_chunkers.encoders import LocalEncoder, OpenAIEncoder

__all__ = [
    "BaseChunker",
    "ConsecutiveChunker",
    "CumulativeChunker",
    "StatisticalChunker",
    "RegexSplitter",
    "BaseSplitter",
    "RegexChunker",
    "CallableEncoder",
    "DenseEncoder",
    "EncoderError",
    "LocalEncoder",
    "OpenAIEncoder",
]

# Imported on first use, not at import time: OpenAIEncoder brings httpx and
# LocalEncoder brings torch, and a user wants at most one of them (PEP 562).
_LAZY_ENCODERS = ("LocalEncoder", "OpenAIEncoder")


def __getattr__(name: str) -> Any:
    if name in _LAZY_ENCODERS:
        from semantic_chunkers import encoders

        return getattr(encoders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(__all__)


try:
    __version__ = version("semantic-chunkers")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"
