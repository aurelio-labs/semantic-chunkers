from importlib import import_module
from typing import TYPE_CHECKING, Any

from semantic_chunkers.encoders.base import (
    CallableEncoder,
    DenseEncoder,
    EncoderError,
    acall_encoder,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers, not for runtime
    from semantic_chunkers.encoders.local import LocalEncoder
    from semantic_chunkers.encoders.openai import OpenAIEncoder

__all__ = [
    "CallableEncoder",
    "DenseEncoder",
    "EncoderError",
    "LocalEncoder",
    "OpenAIEncoder",
    "acall_encoder",
]

# OpenAIEncoder brings httpx and LocalEncoder brings torch. Almost nobody wants
# both, so neither is imported until it is named (PEP 562).
_LAZY = {
    "LocalEncoder": "semantic_chunkers.encoders.local",
    "OpenAIEncoder": "semantic_chunkers.encoders.openai",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(__all__)
