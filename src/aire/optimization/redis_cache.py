"""Redis cache backend for exact-match generation caching (lazy ``aire[redis]``)."""

from __future__ import annotations

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
    """Thin sync Redis client wrapper used by :class:`RedisCachedModel`."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "aire:cache:",
        client: Any | None = None,
    ) -> None:
        redis = _require_redis()
        self.prefix = prefix
        self._client = client or redis.Redis.from_url(url, decode_responses=True)
        self.url = url

    def get(self, key: str) -> str | None:
        value = self._client.get(self.prefix + key)
        return str(value) if value is not None else None

    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        full = self.prefix + key
        if ttl_seconds is not None:
            self._client.setex(full, int(ttl_seconds), value)
        else:
            self._client.set(full, value)

    def delete(self, key: str) -> None:
        self._client.delete(self.prefix + key)

    def ping(self) -> bool:
        return bool(self._client.ping())

    def describe(self) -> dict[str, Any]:
        return {"kind": "redis_cache", "url": self.url, "prefix": self.prefix}


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
        raw = self.backend.get(key)
        if raw is not None:
            self.hits += 1
            return GenerationResult.model_validate_json(raw)
        self.misses += 1
        result = await self.inner.generate(request)
        self.backend.set(
            key,
            result.model_dump_json(),
            ttl_seconds=self.ttl_seconds,
        )
        return result

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        async for chunk in self.inner.stream(request):
            yield chunk

    async def health(self) -> HealthStatus:
        try:
            ok = self.backend.ping()
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
        # SCAN avoids KEYS on large DBs
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=200)
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
