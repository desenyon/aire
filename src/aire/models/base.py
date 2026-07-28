"""Universal model interfaces.

``Model`` and ``EmbeddingModel`` are abstract base classes (structurally
compatible with protocols) that every provider adapter implements. All network
operations are async; sync conveniences live on the facade layer.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

from aire.core.content import Message
from aire.core.errors import OutputValidationError
from aire.core.types import HealthStatus, Manifest, Usage
from aire.models.types import (
    EmbeddingRequest,
    EmbeddingResult,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
)


class Model(abc.ABC):
    """Unified generative model interface."""

    @property
    @abc.abstractmethod
    def info(self) -> ModelInfo:
        """Normalized metadata describing this model."""

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce one completion."""

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Any]:
        """Stream a completion. Default: single-chunk wrapper around generate()."""
        result = await self.generate(request)
        from aire.models.types import GenerationChunk

        yield GenerationChunk(text=result.text, finish_reason=result.finish_reason)

    async def health(self) -> HealthStatus:
        """Liveness probe. Default: attempt a minimal generation."""
        started = time.perf_counter()
        try:
            await self.generate(GenerationRequest.of("ping", max_tokens=1))
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy(latency_ms=(time.perf_counter() - started) * 1000.0)

    # -- conveniences ---------------------------------------------------------------

    async def ask(self, prompt: str | Message | list[Message], **kwargs: Any) -> str:
        """One-shot text-in, text-out convenience."""
        result = await self.generate(GenerationRequest.of(prompt, **kwargs))
        return result.text

    async def ask_structured(
        self,
        prompt: str,
        schema_model: type[Any],
        *,
        retries: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Generate and validate JSON output against a pydantic model."""
        from pydantic import BaseModel

        if not (isinstance(schema_model, type) and issubclass(schema_model, BaseModel)):
            raise OutputValidationError("schema_model must be a pydantic BaseModel subclass")
        from aire.models.types import StructuredOutputSpec

        spec = StructuredOutputSpec(
            name=schema_model.__name__, schema=schema_model.model_json_schema()
        )
        last_error: Exception | None = None
        messages = GenerationRequest.of(prompt, response_format=spec, **kwargs)
        for attempt in range(retries + 1):
            result = await self.generate(messages)
            try:
                return result.parsed(schema_model)
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    repair = Message.text(
                        "user",
                        f"Your previous output failed validation: {exc}. "
                        "Return only valid JSON matching the schema.",
                    )
                    messages = messages.with_messages([*messages.messages, result.message, repair])
        raise OutputValidationError(
            f"model output failed validation after {retries + 1} attempts: {last_error}",
            context={"schema": schema_model.__name__},
            cause=last_error,
        )

    def describe(self) -> Manifest:
        info = self.info
        return Manifest(
            kind="model",
            name=info.ref,
            provider=info.provider,
            capabilities=[str(c) for c in info.capabilities],
            extra={"context_window": info.context_window},
        )


class EmbeddingModel(abc.ABC):
    """Unified embedding model interface."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int: ...

    @abc.abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed a batch of texts."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = await self.embed(EmbeddingRequest(inputs=texts))
        return result.vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def health(self) -> HealthStatus:
        try:
            await self.embed_texts(["ping"])
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(
            kind="embedder",
            name=self.name,
            capabilities=["embeddings"],
            extra={"dimension": self.dimension},
        )


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 chars/token) used when providers don't report usage."""
    return max(1, len(text) // 4)


T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code (facade ``*_sync`` helpers)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("cannot use the sync API inside a running event loop; await instead")


ModelFactory = Callable[..., Model]
EmbedderFactory = Callable[..., EmbeddingModel]
UsageFactory = Callable[[int, int], Usage]
