"""A sentence-transformers encoder, behind the ``local`` extra.

The model runs on your machine, so there is no key and no per-call cost. It is
an extra rather than a core dependency because torch and the model weights are
roughly 2 GB.

```python
from semantic_chunkers import LocalEncoder, StatisticalChunker

chunker = StatisticalChunker(encoder=LocalEncoder())  # pip install semantic-chunkers[local]
```
"""

import asyncio
from typing import Any, List, Optional

import numpy as np

from semantic_chunkers.encoders.base import EncoderError

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEncoder:
    """Embeddings from a local sentence-transformers model.

    :param name: any model name sentence-transformers can load.
    :param device: ``"cpu"``, ``"cuda"``, and so on. Defaults to whatever
        sentence-transformers picks.
    :param batch_size: documents per forward pass.
    :param normalize: return unit vectors. On by default, because the chunkers
        compare with cosine similarity.
    :param score_threshold: read by ``StatisticalChunker`` when
        ``dynamic_threshold=False``. Optional, and unused otherwise.
    """

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        score_threshold: Optional[float] = None,
    ):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.score_threshold = score_threshold
        self._model: Any = None

    def __call__(self, docs: List[str]) -> np.ndarray:
        """Embed ``docs``. The model loads on the first call, not at import."""
        return self.model().encode(
            docs,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    async def acall(self, docs: List[str]) -> np.ndarray:
        """Embed ``docs`` in a worker thread, so the event loop keeps running."""
        return await asyncio.to_thread(self.__call__, docs)

    def model(self) -> Any:
        """The loaded ``SentenceTransformer``, loading it if this is the first ask."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EncoderError(
                    "LocalEncoder needs sentence-transformers: "
                    "pip install 'semantic-chunkers[local]'"
                ) from error
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model
