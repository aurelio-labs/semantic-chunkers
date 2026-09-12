from importlib.metadata import PackageNotFoundError, version

from semantic_chunkers.chunkers import (
    BaseChunker,
    ConsecutiveChunker,
    CumulativeChunker,
    RegexChunker,
    StatisticalChunker,
)
from semantic_chunkers.splitters import BaseSplitter, RegexSplitter

__all__ = [
    "BaseChunker",
    "ConsecutiveChunker",
    "CumulativeChunker",
    "StatisticalChunker",
    "RegexSplitter",
    "BaseSplitter",
    "RegexChunker",
]

try:
    __version__ = version("semantic-chunkers")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"
