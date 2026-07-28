"""Agent memory: short-term buffers and durable JSONL persistence."""

from __future__ import annotations

import abc
from pathlib import Path

from aire.core.content import Message
from aire.core.serialization import iter_jsonl, write_jsonl


class Memory(abc.ABC):
    """Interface for agent memory systems."""

    @abc.abstractmethod
    async def add(self, message: Message) -> None: ...

    @abc.abstractmethod
    async def recall(self, *, limit: int | None = None) -> list[Message]: ...

    @abc.abstractmethod
    async def clear(self) -> None: ...

    def describe(self) -> dict[str, object]:
        return {"kind": "memory", "type": type(self).__name__}


class BufferMemory(Memory):
    """In-memory sliding window of the most recent messages."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._messages: list[Message] = []

    async def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.window:
            del self._messages[: len(self._messages) - self.window]

    async def recall(self, *, limit: int | None = None) -> list[Message]:
        messages = self._messages[-limit:] if limit else list(self._messages)
        return messages

    async def clear(self) -> None:
        self._messages.clear()


class JsonlMemory(Memory):
    """Durable append-only memory backed by a JSONL file."""

    def __init__(self, path: str | Path, *, window: int = 500) -> None:
        self.path = Path(path)
        self.window = window
        self._buffer = BufferMemory(window=window)
        if self.path.is_file():
            for row in iter_jsonl(self.path):
                self._buffer._messages.append(Message.model_validate(row))

    async def add(self, message: Message) -> None:
        await self._buffer.add(message)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(message.model_dump_json() + "\n")

    async def recall(self, *, limit: int | None = None) -> list[Message]:
        return await self._buffer.recall(limit=limit)

    async def clear(self) -> None:
        await self._buffer.clear()
        write_jsonl(self.path, [])


_MEMORY_KINDS: dict[str, type[Memory]] = {
    "buffer": BufferMemory,
    "jsonl": JsonlMemory,
}


def resolve_memory(spec: str | Memory | None) -> Memory:
    """Resolve a memory spec like ``"buffer"``, ``"jsonl:path"``, ``"long-term"`` or an instance."""
    if isinstance(spec, Memory):
        return spec
    if spec is None or spec == "buffer":
        return BufferMemory()
    if spec.startswith("jsonl:"):
        return JsonlMemory(spec.split(":", 1)[1])
    if spec == "long-term" or spec.startswith("long-term:"):
        from aire.memory.store import LongTermMemory

        path = spec.split(":", 1)[1] if ":" in spec else None
        return LongTermMemory(path=path or None)
    if spec in _MEMORY_KINDS:
        return _MEMORY_KINDS[spec]()
    from aire.core.errors import NotFoundError

    raise NotFoundError(
        "memory",
        spec,
        context={"available": sorted([*_MEMORY_KINDS, "long-term", "jsonl:<path>"])},
    )
