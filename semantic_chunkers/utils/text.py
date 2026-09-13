import asyncio
import time
from functools import wraps

import tiktoken

from semantic_chunkers.utils.logger import logger


def tiktoken_length(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def time_it(func):
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # Await the async function
        end_time = time.time()
        logger.debug(f"{func.__name__} duration: {end_time - start_time:.2f} seconds")
        return result

    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the sync function directly
        end_time = time.time()
        logger.debug(f"{func.__name__} duration: {end_time - start_time:.2f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def async_retry_with_timeout(retries=3, timeout=10):
    """Retry an async function, giving each attempt ``timeout`` seconds.

    The failure that ends the last attempt is raised, timeout included. A
    decorator that swallowed it would return ``None``, and the caller would
    fail somewhere further on with nothing left to say why.
    """
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                last_attempt = attempt == retries - 1
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout on attempt {attempt + 1} for {func.__name__}"
                    )
                    if last_attempt:
                        raise
                except Exception as e:
                    logger.error(
                        f"Exception on attempt {attempt + 1} for {func.__name__}: {e}"
                    )
                    if last_attempt:
                        raise
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return wrapper

    return decorator
