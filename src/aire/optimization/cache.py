"""Response caching: exact-match and semantic caching model wrappers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from typing import Any

from aire.core.types import HealthStatus
from aire.models.base import EmbeddingModel, Model
from aire.models.types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
)
from aire.rag.store import cosine_similarity


def _request_key(request: GenerationRequest, model_ref: str) -> str:
    payload = request.model_dump_json(exclude={"metadata"})
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
                return result
            del self._cache[key]
        self.misses += 1
        result = await self.inner.generate(request)
        if len(self._cache) >= self.max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]
        self._cache[key] = (now, result)
        return result

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        async for chunk in self.inner.stream(request):
            yield chunk

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


class SemanticCachedModel(Model):
    """Caches by embedding similarity: near-duplicate prompts hit the cache."""

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
        self._entries: list[tuple[list[float], str, GenerationResult]] = []
        self.hits = 0
        self.misses = 0

    @property
    def info(self) -> ModelInfo:
        return self.inner.info

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = "\n".join(m.text_content for m in request.messages)
        vector = await self.embedder.embed_one(prompt)
        for cached_vector, _cached_prompt, result in self._entries:
            if cosine_similarity(vector, cached_vector) >= self.threshold:
                self.hits += 1
                return result
        self.misses += 1
        result = await self.inner.generate(request)
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append((vector, prompt, result))
        return result

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
