"""Resource lifecycle management.

Anything holding external state (HTTP clients, connections, temp files)
registers with the :class:`ResourceManager`; closing the runtime releases
everything in reverse acquisition order, guaranteeing safe shutdown even on
partial failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from aire.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class AsyncResource(Protocol):
    async def aclose(self) -> None: ...


CleanupFn = Callable[[], Any | Awaitable[Any]]
T = TypeVar("T")


class ResourceManager:
    """Tracks open resources and closes them LIFO."""

    def __init__(self) -> None:
        self._cleanups: list[tuple[str, CleanupFn]] = []
        self._closed = False

    def track(self, name: str, cleanup: CleanupFn) -> None:
        if self._closed:
            raise RuntimeError(f"resource manager already closed; cannot track {name!r}")
        self._cleanups.append((name, cleanup))

    def track_resource(self, name: str, resource: T) -> T:
        """Track an object exposing aclose()/close() and return it unchanged."""
        aclose = getattr(resource, "aclose", None)
        if callable(aclose):
            self.track(name, aclose)
        else:
            close = getattr(resource, "close", None)
            if callable(close):
                self.track(name, close)
        return resource

    @property
    def open_count(self) -> int:
        return len(self._cleanups)

    async def aclose(self) -> None:
        errors: list[str] = []
        while self._cleanups:
            name, cleanup = self._cleanups.pop()
            try:
                result = cleanup()
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    await result
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.warning("resource_close_failed", resource=name, error=str(exc))
        self._closed = True
        if errors:
            logger.warning("resource_close_errors", errors=errors)

    def close(self) -> None:
        """Synchronous close for non-async callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            raise RuntimeError("use aclose() inside a running event loop")

    async def __aenter__(self) -> ResourceManager:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()
