"""Redis cache backend for exact-match generation caching (lazy ``aire[redis]``)."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import HealthStatus
from aire.models.base import Model
from aire.models.types import GenerationChunk, GenerationRequest, GenerationResult, ModelInfo
from aire.optimization.cache import cache_key


def _require_redis() -> Any:
    if importlib.util.find_spec("redis") is None:
        raise ConfigurationError(
            "redis is required for RedisCachedModel: pip install 'aire[redis]'",
            code="optimization.redis_missing",
            context={"extra": "aire[redis]", "package": "redis"},
        )
    import redis  # type: ignore[import-not-found]

    return redis


class RedisCacheBackend:
    """Redis client wrapper used by :class:`RedisCachedModel`.

    Prefers ``redis.asyncio`` when available; falls back to sync Redis via
    ``asyncio.to_thread``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "aire:cache:",
        client: Any | None = None,
    ) -> None:
        redis = _require_redis()
        self.prefix = prefix
        self.url = url
        self._async = False
        if client is not None:
            self._client = client
            self._async = hasattr(client, "get") and asyncio.iscoroutinefunction(
                getattr(client, "get", None)
            )
        else:
            async_mod = getattr(redis, "asyncio", None)
            if async_mod is not None and hasattr(async_mod, "Redis"):
                self._client = async_mod.Redis.from_url(url, decode_responses=True)
                self._async = True
            else:
                self._client = redis.Redis.from_url(url, decode_responses=True)

    async def aget(self, key: str) -> str | None:
        full = self.prefix + key
        if self._async:
            value = await self._client.get(full)
        else:
            value = await asyncio.to_thread(self._client.get, full)
        return str(value) if value is not None else None

    async def aset(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        full = self.prefix + key
        if self._async:
            if ttl_seconds is not None:
                await self._client.setex(full, int(ttl_seconds), value)
            else:
                await self._client.set(full, value)
            return
        if ttl_seconds is not None:
            await asyncio.to_thread(self._client.setex, full, int(ttl_seconds), value)
        else:
            await asyncio.to_thread(self._client.set, full, value)

    def get(self, key: str) -> str | None:
        value = self._client.get(self.prefix + key)
        # Sync path only — async clients must use aget
        if asyncio.iscoroutine(value):
            raise ConfigurationError(
                "async Redis client requires aget(); use RedisCachedModel.generate",
                code="optimization.redis_async",
            )
        return str(value) if value is not None else None

    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        full = self.prefix + key
        if ttl_seconds is not None:
            result = self._client.setex(full, int(ttl_seconds), value)
        else:
            result = self._client.set(full, value)
        if asyncio.iscoroutine(result):
            raise ConfigurationError(
                "async Redis client requires aset(); use RedisCachedModel.generate",
                code="optimization.redis_async",
            )

    def delete(self, key: str) -> None:
        self._client.delete(self.prefix + key)

    def ping(self) -> bool:
        result = self._client.ping()
        if asyncio.iscoroutine(result):
            raise ConfigurationError(
                "async Redis client: use RedisCachedModel.health()",
                code="optimization.redis_async",
            )
        return bool(result)

    async def aping(self) -> bool:
        if self._async:
            return bool(await self._client.ping())
        return bool(await asyncio.to_thread(self._client.ping))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "redis_cache",
            "url": self.url,
            "prefix": self.prefix,
            "async": self._async,
        }


class RedisCachedModel(Model):
    """Exact-match response cache backed by Redis (lazy import)."""

    def __init__(
        self,
        inner: Model,
        backend: RedisCacheBackend | str | None = None,
        *,
        ttl_seconds: float | None = 3600,
    ) -> None:
        self.inner = inner
        if isinstance(backend, RedisCacheBackend):
            self.backend = backend
        elif isinstance(backend, str):
            self.backend = RedisCacheBackend(backend)
        else:
            self.backend = RedisCacheBackend()
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    @property
    def info(self) -> ModelInfo:
        return self.inner.info

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        key = cache_key(request, self.inner.info.ref)
        raw = await self.backend.aget(key)
        if raw is not None:
            self.hits += 1
            return GenerationResult.model_validate_json(raw)
        self.misses += 1
        result = await self.inner.generate(request)
        await self.backend.aset(
            key,
            result.model_dump_json(),
            ttl_seconds=self.ttl_seconds,
        )
        return result

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Stream then cache the full completion (same contract as CachedModel)."""
        key = cache_key(request, self.inner.info.ref)
        raw = await self.backend.aget(key)
        if raw is not None:
            self.hits += 1
            result = GenerationResult.model_validate_json(raw)
            yield GenerationChunk(
                text=result.text, finish_reason=result.finish_reason, usage=result.usage
            )
            return
        self.misses += 1
        pieces: list[str] = []
        finish: str | None = None
        last_usage = None
        async for chunk in self.inner.stream(request):
            if chunk.text:
                pieces.append(chunk.text)
            if chunk.finish_reason:
                finish = chunk.finish_reason
            if chunk.usage is not None:
                last_usage = chunk.usage
            yield chunk
        text = "".join(pieces)
        result = GenerationResult.text_result(
            text, model=self.inner.info.ref, usage=last_usage
        )
        if finish:
            result = result.model_copy(update={"finish_reason": finish})
        await self.backend.aset(
            key, result.model_dump_json(), ttl_seconds=self.ttl_seconds
        )

    async def health(self) -> HealthStatus:
        try:
            ok = await self.backend.aping()
        except Exception as exc:
            return HealthStatus.unhealthy(f"redis: {exc}")
        if not ok:
            return HealthStatus.unhealthy("redis ping failed")
        return await self.inner.health()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "backend": "redis",
            "checked_at": time.time(),
        }

    def clear(self) -> int:
        """Delete all keys under this cache prefix. Returns number deleted."""
        pattern = self.backend.prefix + "*"
        client = self.backend._client
        deleted = 0
        cursor = 0
        while True:
            scan_result = client.scan(cursor=cursor, match=pattern, count=200)
            if asyncio.iscoroutine(scan_result):
                raise ConfigurationError(
                    "clear() on async Redis requires an event loop; use sync client",
                    code="optimization.redis_async",
                )
            cursor, keys = scan_result
            if keys:
                deleted += int(client.delete(*keys))
            if cursor == 0:
                break
        self.hits = 0
        self.misses = 0
        return deleted


def describe() -> dict[str, Any]:
    return {
        "kind": "redis_cache",
        "available": importlib.util.find_spec("redis") is not None,
        "install": "pip install 'aire[redis]'",
    }
