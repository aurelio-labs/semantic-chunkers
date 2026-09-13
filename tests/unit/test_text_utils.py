"""The retry helper the async encoder runs behind."""

import asyncio

import pytest

from semantic_chunkers.utils.text import async_retry_with_timeout


@pytest.mark.asyncio
async def test_async_retry_with_timeout_raises_when_every_attempt_stalls():
    """A stall that uses up the retries is raised, not returned as None."""
    attempts = 0

    @async_retry_with_timeout(retries=2, timeout=0.01)
    async def stall():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        await stall()
    assert attempts == 2


@pytest.mark.asyncio
async def test_async_retry_with_timeout_returns_after_a_stalled_attempt():
    attempts = 0

    @async_retry_with_timeout(retries=3, timeout=0.05)
    async def stall_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(10)
        return "embeddings"

    assert await stall_once() == "embeddings"
    assert attempts == 2


@pytest.mark.asyncio
async def test_async_retry_with_timeout_raises_the_last_exception():
    @async_retry_with_timeout(retries=1, timeout=1)
    async def refuse():
        raise RuntimeError("the encoder said no")

    with pytest.raises(RuntimeError, match="the encoder said no"):
        await refuse()


def test_async_retry_with_timeout_rejects_a_budget_of_no_attempts():
    with pytest.raises(ValueError):
        async_retry_with_timeout(retries=0)
