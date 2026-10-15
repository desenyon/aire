"""Response caching: exact-match and semantic caching model wrappers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from typing import Any

from aire.core.content import TextContent
from aire.core.types import HealthStatus, Usage
from aire.models.base import EmbeddingModel, Model
from aire.models.types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ToolCall,
)
from aire.rag.store import cosine_similarity


def _request_key(request: GenerationRequest, model_ref: str) -> str:
    payload = request.model_dump_json()
    return hashlib.sha256(f"{model_ref}|{payload}".encode()).hexdigest()


class CachedModel(Model):
    """Wraps a model with an exact-match response cache."""

    def __init__(
        self, inner: Model, *, ttl_seconds: float | None = None, max_entries: int = 1024
    ) -> None:
        self.inner = inner
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, tuple[float, GenerationResult]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def info(self) -> ModelInfo:
        return self.inner.info

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        key = _request_key(request, self.inner.info.ref)
        now = time.time()
        if key in self._cache:
            created, result = self._cache[key]
            if self.ttl_seconds is None or now - created < self.ttl_seconds:
                self.hits += 1
                # Deep copy: callers must never mutate another caller's result.
                return replay_result(result)
            del self._cache[key]
        self.misses += 1
        result = await self.inner.generate(request)
        if len(self._cache) >= self.max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]
        # Store a copy: mutating the returned result must not poison the cache.
        self._cache[key] = (now, result.model_copy(deep=True))
        return result

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Stream from the inner model, then cache the full result.

        Cache lookup on stream hits replays as a single chunk. Streaming does
        not partially cache mid-flight.
        """
        key = _request_key(request, self.inner.info.ref)
        now = time.time()
        if key in self._cache:
            created, result = self._cache[key]
            if self.ttl_seconds is None or now - created < self.ttl_seconds:
                self.hits += 1
                yield GenerationChunk(
                    text=result.text,
                    tool_calls=[call.model_copy(deep=True) for call in result.tool_calls],
                    finish_reason=result.finish_reason,
                    usage=Usage(),
                )
                return
            del self._cache[key]
        self.misses += 1
        pieces: list[str] = []
        calls: list[ToolCall] = []
        finish: str | None = None
        last_usage = None
        async for chunk in self.inner.stream(request):
            calls.extend(call.model_copy(deep=True) for call in chunk.tool_calls)
            pieces.append(chunk.text)
            finish = chunk.finish_reason or finish
            last_usage = chunk.usage if chunk.usage is not None else last_usage
            yield chunk
        text = "".join(pieces)
        result = GenerationResult.text_result(
            text, model=self.inner.info.ref, usage=last_usage, tool_calls=calls
        )
        if finish:
            result = result.model_copy(update={"finish_reason": finish})
        if len(self._cache) >= self.max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]
        self._cache[key] = (now, result.model_copy(deep=True))

    async def health(self) -> HealthStatus:
        return await self.inner.health()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "entries": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()


def _has_tool_context(request: GenerationRequest) -> bool:
    return (
        bool(request.tools)
        or not request.messages
        or request.messages[-1].role != "user"
        or any(
            m.tool_calls
            or m.role == "tool"
            or any(not isinstance(c, TextContent) for c in m.content)
            for m in request.messages
        )
    )


def _params_signature(request: GenerationRequest) -> str:
    """Exact signature of generation-affecting parameters (everything but the
    messages/metadata). Semantic hits require this to match — a structured
    output request must never be served a plain-text cache entry."""
    # Only the final user text is eligible for semantic comparison. Preserve
    # all preceding instructions/history and caller namespaces exactly.
    payload_data = request.model_dump(mode="json", exclude_none=True)
    if payload_data.get("messages"):
        payload_data["messages"][-1]["content"] = []
    import json

    payload = json.dumps(payload_data, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class SemanticCachedModel(Model):
    """Caches by embedding similarity: near-duplicate prompts hit the cache,
    but only when generation parameters (temperature, response_format, ...)
    match exactly."""

    def __init__(
        self,
        inner: Model,
        embedder: EmbeddingModel,
        *,
        threshold: float = 0.95,
        max_entries: int = 1024,
    ) -> None:
        self.inner = inner
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: list[tuple[str, list[float], str, GenerationResult]] = []
        self.hits = 0
        self.misses = 0

    @property
    def info(self) -> ModelInfo:
        return self.inner.info

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if _has_tool_context(request):
            self.misses += 1
            return await self.inner.generate(request)
        prompt = "\n".join(m.text_content for m in request.messages)
        signature = _params_signature(request)
        vector = await self.embedder.embed_one(prompt)
        for cached_signature, cached_vector, _cached_prompt, result in self._entries:
            if cached_signature != signature:
                continue
            if cosine_similarity(vector, cached_vector) >= self.threshold:
                self.hits += 1
                return replay_result(result)
        self.misses += 1
        result = await self.inner.generate(request)
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append((signature, vector, prompt, result.model_copy(deep=True)))
        return result

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Stream then cache the full completion; hits replay as one chunk.

        Semantic + stream caching only stores after the stream completes.
        """
        if _has_tool_context(request):
            self.misses += 1
            async for chunk in self.inner.stream(request):
                yield chunk
            return
        prompt = "\n".join(m.text_content for m in request.messages)
        signature = _params_signature(request)
        vector = await self.embedder.embed_one(prompt)
        for cached_signature, cached_vector, _cached_prompt, result in self._entries:
            if cached_signature != signature:
                continue
            if cosine_similarity(vector, cached_vector) >= self.threshold:
                self.hits += 1
                yield GenerationChunk(
                    text=result.text,
                    tool_calls=[call.model_copy(deep=True) for call in result.tool_calls],
                    finish_reason=result.finish_reason,
                    usage=Usage(),
                )
                return
        self.misses += 1
        pieces: list[str] = []
        calls: list[ToolCall] = []
        finish: str | None = None
        last_usage = None
        async for chunk in self.inner.stream(request):
            calls.extend(call.model_copy(deep=True) for call in chunk.tool_calls)
            pieces.append(chunk.text)
            finish = chunk.finish_reason or finish
            last_usage = chunk.usage if chunk.usage is not None else last_usage
            yield chunk
        text = "".join(pieces)
        result = GenerationResult.text_result(
            text, model=self.inner.info.ref, usage=last_usage, tool_calls=calls
        )
        if finish:
            result = result.model_copy(update={"finish_reason": finish})
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append((signature, vector, prompt, result.model_copy(deep=True)))

    async def health(self) -> HealthStatus:
        return await self.inner.health()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "entries": len(self._entries),
            "threshold": self.threshold,
        }

    def clear(self) -> None:
        self._entries.clear()


def cache_key(request: GenerationRequest, model_ref: str) -> str:
    """Public helper: the exact cache key for a request."""
    return _request_key(request, model_ref)


def replay_result(result: GenerationResult) -> GenerationResult:
    """Replay content without charging an already completed generation again."""
    replay = result.model_copy(deep=True)
    return replay.model_copy(
        update={
            "usage": Usage(),
            "raw": {
                **(replay.raw or {}),
                "cache_hit": True,
                "source_usage": result.usage.model_dump(),
            },
        }
    )
