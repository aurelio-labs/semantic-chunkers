"""The encoder protocol every chunker works against.

A chunker asks an encoder for one thing: hand it a list of documents, get back
one vector per document. That is small enough to be a ``Protocol`` rather than
a base class, so anything with the right two methods works — including the
Semantic Router encoders this package used to import, which keep working by duck
typing.
"""

import asyncio
from typing import Any, List, Protocol, Union, runtime_checkable

import numpy as np

# What an encoder hands back: a numpy array, or the list of lists that most
# HTTP encoders decode from JSON. The chunkers accept either.
Embeddings = Union[np.ndarray, List[List[float]]]


class EncoderError(RuntimeError):
    """Raised when an encoder cannot do what a chunker asked of it."""


@runtime_checkable
class DenseEncoder(Protocol):
    """One vector per document, synchronously and asynchronously.

    ``docs`` is a list because encoders batch; it is ``Any`` rather than ``str``
    because ``ConsecutiveChunker`` also chunks sequences of video frames.

    ```python
    class MyEncoder:
        def __call__(self, docs): return model.encode(docs)
        async def acall(self, docs): return await model.aencode(docs)
    ```

    An encoder with no async path still works on the synchronous call; the
    chunkers raise :class:`EncoderError` if you reach for ``acall`` without one.
    Wrap a plain callable in :class:`CallableEncoder` to get both.
    """

    def __call__(self, docs: List[Any]) -> Embeddings: ...

    async def acall(self, docs: List[Any]) -> Embeddings: ...


class CallableEncoder:
    """Adapt any ``docs -> vectors`` callable to :class:`DenseEncoder`.

    The callable supplies ``__call__``; ``acall`` runs it in a worker thread, so
    a synchronous encoder does not block the event loop.

    ```python
    from semantic_chunkers import CallableEncoder, StatisticalChunker

    encoder = CallableEncoder(lambda docs: model.encode(docs), name="my-model")
    chunker = StatisticalChunker(encoder=encoder)
    ```
    """

    def __init__(
        self, encode: Any, name: str = "callable", score_threshold: Any = None
    ):
        if not callable(encode):
            raise EncoderError(
                f"CallableEncoder needs a callable, got {type(encode).__name__}."
            )
        self.encode = encode
        self.name = name
        self.score_threshold = score_threshold

    def __call__(self, docs: List[Any]) -> Embeddings:
        return self.encode(docs)

    async def acall(self, docs: List[Any]) -> Embeddings:
        return await asyncio.to_thread(self.encode, docs)


async def acall_encoder(encoder: Any, docs: List[Any]) -> Embeddings:
    """Encode ``docs`` asynchronously, failing with a message that names the encoder.

    Two encoders have no async path: one with no ``acall`` at all, and one that
    inherits a stub raising ``NotImplementedError`` from a file the caller has
    never opened (issue #32). Both arrive here as the same instruction.
    """
    acall = getattr(encoder, "acall", None)
    if acall is None:
        raise EncoderError(_no_async_path(encoder))
    try:
        return await acall(docs)
    except NotImplementedError as error:
        raise EncoderError(_no_async_path(encoder)) from error


def _no_async_path(encoder: Any) -> str:
    name = type(encoder).__name__
    return (
        f"{name} has no usable acall(), so this chunker cannot run asynchronously. "
        f"Give it an `async def acall(self, docs)`, call the chunker synchronously, "
        f"or wrap it: CallableEncoder(encoder) runs the synchronous call in a thread."
    )
