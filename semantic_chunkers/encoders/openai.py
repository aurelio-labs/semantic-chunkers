"""The OpenAI embeddings endpoint, called over ``httpx``.

First-party and in the core, so ``pip install semantic-chunkers`` gives a
working encoder with no extra and without the ``openai`` SDK. ``base_url`` is
configurable, so Azure and any OpenAI-compatible endpoint work through the same
class.

```python
from semantic_chunkers import OpenAIEncoder, StatisticalChunker

encoder = OpenAIEncoder()  # reads OPENAI_API_KEY
chunker = StatisticalChunker(encoder=encoder)
```
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from semantic_chunkers.encoders.base import EncoderError

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# The endpoint accepts far more, but a smaller batch keeps one failure cheap.
DEFAULT_BATCH_SIZE = 100
FIRST_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 20.0


class OpenAIEncoderError(EncoderError):
    """Raised when the embeddings endpoint returns an error.

    The message carries the API's own ``error.message`` when it sent one, so the
    reason ("invalid api key", "model not found") is in the traceback.
    """


class OpenAIEncoder:
    """Embeddings from OpenAI or any OpenAI-compatible endpoint.

    :param name: the embedding model.
    :param api_key: falls back to ``OPENAI_API_KEY``. Required, as the OpenAI
        SDK requires it; pass any placeholder for an endpoint that ignores it.
    :param base_url: falls back to ``OPENAI_BASE_URL``, then to OpenAI's own.
    :param org_id: sent as ``OpenAI-Organization`` when given.
    :param dimensions: truncate the vectors, for models that support it.
    :param batch_size: documents per request.
    :param max_retries: retries for 429 and 5xx, backing off exponentially and
        honouring ``Retry-After``. Other 4xx raise immediately.
    :param timeout: seconds per request.
    :param score_threshold: read by ``StatisticalChunker`` when
        ``dynamic_threshold=False``. Optional, and unused otherwise.
    :param transport: an ``httpx`` transport, for tests and custom HTTP setups.
    """

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        org_id: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        timeout: float = 30.0,
        score_threshold: Optional[float] = None,
        transport: Optional[Any] = None,
    ):
        self.name = name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAIEncoderError(
                "No OpenAI API key. Pass api_key= or set OPENAI_API_KEY."
            )
        base = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.org_id = org_id or os.environ.get("OPENAI_ORG_ID")
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.score_threshold = score_threshold
        self._transport = transport
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_client_loop: Optional[asyncio.AbstractEventLoop] = None

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Embed ``docs``, one request per ``batch_size`` documents."""
        vectors: List[List[float]] = []
        for batch in self._batches(docs):
            vectors.extend(self._embed(batch))
        return vectors

    async def acall(self, docs: List[str]) -> List[List[float]]:
        """Embed ``docs`` asynchronously. Batches go out concurrently."""
        batches = self._batches(docs)
        results = await asyncio.gather(*(self._aembed(batch) for batch in batches))
        return [vector for batch_vectors in results for vector in batch_vectors]

    def close(self) -> None:
        """Close the HTTP connection pool. Optional; it closes on collection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close the async HTTP connection pool."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_client_loop = None

    def _batches(self, docs: List[str]) -> List[List[str]]:
        return [
            docs[start : start + self.batch_size]
            for start in range(0, len(docs), self.batch_size)
        ]

    def _embed(self, batch: List[str]) -> List[List[float]]:
        client = self._sync_client()
        attempt = 0
        while True:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=self._payload(batch),
            )
            if attempt >= self.max_retries or not _retryable(response):
                return _vectors(response)
            time.sleep(_backoff(response, attempt))
            attempt += 1

    async def _aembed(self, batch: List[str]) -> List[List[float]]:
        client = self._get_async_client()
        attempt = 0
        while True:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=self._payload(batch),
            )
            if attempt >= self.max_retries or not _retryable(response):
                return _vectors(response)
            await asyncio.sleep(_backoff(response, attempt))
            attempt += 1

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        # A connection pool belongs to the loop that opened it, so a second
        # asyncio.run() has to start a fresh client rather than reuse the first.
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_client_loop is not loop:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout, transport=self._transport
            )
            self._async_client_loop = loop
        return self._async_client

    def _headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        return headers

    def _payload(self, batch: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"input": batch, "model": self.name}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload


def _retryable(response: httpx.Response) -> bool:
    """Rate limits and server faults are worth another go; other errors are not."""
    return response.status_code == 429 or response.status_code >= 500


def _backoff(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            # Retry-After may be an HTTP date instead of seconds; back off instead.
            pass
    return min(FIRST_BACKOFF_SECONDS * 2**attempt, MAX_BACKOFF_SECONDS)


def _vectors(response: httpx.Response) -> List[List[float]]:
    if response.status_code >= 400:
        raise OpenAIEncoderError(_error_message(response))
    try:
        body = response.json()
    except ValueError as error:
        raise OpenAIEncoderError(
            f"The embeddings endpoint returned no JSON: {response.text[:200]!r}"
        ) from error
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not all(
        isinstance(item, dict) and "embedding" in item for item in data
    ):
        raise OpenAIEncoderError(
            f"The embeddings endpoint returned no embeddings: {response.text[:200]!r}"
        )
    # The response carries an index per embedding and need not be ordered. An
    # OpenAI-compatible endpoint that omits the index is taken in the order it
    # sent, rather than failing on the missing key.
    ordered = sorted(enumerate(data), key=lambda pair: pair[1].get("index", pair[0]))
    return [item["embedding"] for _, item in ordered]


def _error_message(response: httpx.Response) -> str:
    detail = response.text[:200]
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        detail = body["error"].get("message", detail)
    return f"The embeddings endpoint returned {response.status_code}: {detail}"
