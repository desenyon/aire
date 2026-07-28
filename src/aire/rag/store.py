"""Vector store interface and the zero-dependency local implementation."""

from __future__ import annotations

import abc
import json
import math
import re
from pathlib import Path
from typing import Any

from aire.core.errors import RetrievalError
from aire.core.types import HealthStatus, Manifest
from aire.rag.types import Chunk, ScoredChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for equal-length vectors (pure python, no numpy)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class VectorStore(abc.ABC):
    """Interface every vector store adapter implements."""

    @abc.abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> int:
        """Insert or replace chunks (embeddings must be attached)."""

    @abc.abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Similarity search over stored embeddings."""

    @abc.abstractmethod
    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Keyword search (BM25-style) for hybrid retrieval."""

    @abc.abstractmethod
    async def delete(self, ids: list[str]) -> int: ...

    @abc.abstractmethod
    async def count(self) -> int: ...

    async def clear(self) -> None:
        ids = [c.chunk.id for c in await self.search_text("", k=1_000_000)]
        if ids:
            await self.delete(ids)

    async def health(self) -> HealthStatus:
        try:
            await self.count()
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(kind="vector_store", name=type(self).__name__)


class LocalVectorStore(VectorStore):
    """In-memory vector store with optional JSON persistence.

    This is the default store: zero services, deterministic, fast enough for
    tens of thousands of chunks. Swap in ``qdrant:``/``chroma:`` for scale.
    """

    def __init__(self, path: str | Path | None = None, *, name: str = "local") -> None:
        self._chunks: dict[str, Chunk] = {}
        self._path = Path(path) if path else None
        self._name = name
        if self._path and self._path.is_file():
            self._load()

    # -- persistence -----------------------------------------------------------------

    def _load(self) -> None:
        assert self._path is not None
        try:
            payload = json.loads(self._path.read_text())
            self._chunks = {c["id"]: Chunk.model_validate(c) for c in payload.get("chunks", [])}
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            raise RetrievalError(
                f"corrupt local vector store at {self._path}: {exc}",
                context={"path": str(self._path)},
                cause=exc,
            ) from exc

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self._path
        if target is None:
            raise RetrievalError("no path configured for persistence")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [c.model_dump(mode="json") for c in self._chunks.values()]}
        target.write_text(json.dumps(payload))
        self._path = target
        return target

    # -- interface ----------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return len(chunks)

    def _matches(self, chunk: Chunk, filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        return all(chunk.metadata.get(key) == value for key, value in filter.items())

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=c, score=cosine_similarity(vector, c.embedding or []))
            for c in self._chunks.values()
            if c.embedding is not None and self._matches(c, filter)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        if not query_terms:
            candidates = list(self._chunks.values())
            return [ScoredChunk(chunk=c, score=0.0) for c in candidates[:k]]
        df: dict[str, int] = {}
        docs = {c.id: tokenize(c.text) for c in self._chunks.values() if self._matches(c, filter)}
        for terms in docs.values():
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        n_docs = max(len(docs), 1)
        scored: list[ScoredChunk] = []
        for chunk_id, terms in docs.items():
            score = 0.0
            for term in query_terms:
                tf = terms.count(term)
                if tf == 0:
                    continue
                idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2)
            if score > 0:
                scored.append(ScoredChunk(chunk=self._chunks[chunk_id], score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        removed = 0
        for chunk_id in ids:
            if self._chunks.pop(chunk_id, None) is not None:
                removed += 1
        return removed

    async def count(self) -> int:
        return len(self._chunks)

    async def clear(self) -> None:
        self._chunks.clear()

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self._name,
            provider="local",
            capabilities=["vector-search", "keyword-search", "persistence"],
            extra={"count": len(self._chunks), "path": str(self._path) if self._path else None},
        )


def register(runtime: Any) -> None:
    """Register the local vector store factory on a runtime."""

    def _factory(name: str = "default", *, runtime: Any = None, **options: Any) -> VectorStore:
        return LocalVectorStore(**options)

    runtime.vector_stores.register("local", _factory, replace=True)
