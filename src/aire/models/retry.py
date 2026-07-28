"""Retry helpers with exponential backoff for transient provider failures."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aire.core.errors import AireError

T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (AireError,),
) -> T:
    """Retry ``operation`` while it raises a retryable error.

    An error is retried only if it is an instance of ``retry_on`` and its
    ``retryable`` attribute is truthy (non-AireError exceptions in ``retry_on``
    are always retried).
    """
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:
            last = exc
            retryable = getattr(exc, "retryable", True)
            if not retryable or attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())  # noqa: S311 - jitter, not security
            await asyncio.sleep(delay)
    assert last is not None
    raise last
