"""Synchronous + asynchronous event bus.

Events are structured dict-like payloads with a topic string. Subsystems emit
events (``model.call``, ``tool.execute``, ``agent.step``, ...) and observers
subscribe without the emitter knowing who listens.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aire.core.types import new_id

Handler = Callable[["Event"], Any | Awaitable[Any]]


@dataclass(frozen=True)
class Event:
    """A single occurrence on the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    timestamp: float = field(default_factory=time.time)
    source: str | None = None

    def matches(self, pattern: str) -> bool:
        """Match ``exact.topic`` or ``prefix.*`` wildcard patterns."""
        if pattern.endswith(".*"):
            return self.topic.startswith(pattern[:-1])
        return self.topic == pattern


class EventBus:
    """Fan-out dispatcher with wildcard subscription support."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self.history: list[Event] = []
        self._keep_history = True

    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        """Subscribe a handler; returns an unsubscribe function."""
        self._handlers.setdefault(pattern, []).append(handler)

        def _unsubscribe() -> None:
            handlers = self._handlers.get(pattern, [])
            if handler in handlers:
                handlers.remove(handler)

        return _unsubscribe

    def emit(
        self, topic: str, data: dict[str, Any] | None = None, *, source: str | None = None
    ) -> Event:
        """Emit an event, invoking sync handlers immediately and scheduling async ones."""
        event = Event(topic=topic, data=dict(data or {}), source=source)
        if self._keep_history:
            self.history.append(event)
        for pattern, handlers in list(self._handlers.items()):
            if not event.matches(pattern):
                continue
            for handler in list(handlers):
                result = handler(event)
                if inspect.isawaitable(result):
                    _schedule(result)
        return event

    async def emit_async(
        self, topic: str, data: dict[str, Any] | None = None, *, source: str | None = None
    ) -> Event:
        """Emit an event and await all async handlers."""
        event = Event(topic=topic, data=dict(data or {}), source=source)
        if self._keep_history:
            self.history.append(event)
        tasks: list[Awaitable[Any]] = []
        for pattern, handlers in list(self._handlers.items()):
            if not event.matches(pattern):
                continue
            for handler in list(handlers):
                result = handler(event)
                if inspect.isawaitable(result):
                    tasks.append(result)
        if tasks:
            await asyncio.gather(*tasks)
        return event

    def clear_history(self) -> None:
        self.history.clear()


_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _schedule(awaitable: Awaitable[Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_consume(awaitable))
    else:
        task = loop.create_task(_consume(awaitable))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _consume(awaitable: Awaitable[Any]) -> None:
    await awaitable
