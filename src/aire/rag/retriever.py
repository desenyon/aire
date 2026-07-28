"""Hybrid retrieval: vector similarity + keyword search with RRF fusion."""

from __future__ import annotations

from typing import Any

from aire.models.base import EmbeddingModel
from aire.rag.store import VectorStore
from aire.rag.types import ScoredChunk


class Retriever:
    """Retrieves relevant chunks from a store, optionally fusing keyword results.

    Hybrid fusion uses Reciprocal Rank Fusion (RRF), which is robust to the
    different score scales of vector and keyword search.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingModel,
        *,
        hybrid: bool = True,
        vector_weight: float = 1.0,
        keyword_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.hybrid = hybrid
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        candidates: int | None = None,
    ) -> list[ScoredChunk]:
        candidate_k = candidates or max(k * 4, 10)
        vector = await self.embedder.embed_one(query)
        vector_hits = await self.store.search(vector, k=candidate_k, filter=filter)
        if not self.hybrid:
            return vector_hits[:k]
        keyword_hits = await self.store.search_text(query, k=candidate_k, filter=filter)
        return self._fuse(vector_hits, keyword_hits)[:k]

    def _fuse(
        self, vector_hits: list[ScoredChunk], keyword_hits: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Any] = {}
        for rank, hit in enumerate(vector_hits):
            key = hit.chunk.id
            chunks[key] = hit.chunk
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (self.rrf_k + rank + 1)
        for rank, hit in enumerate(keyword_hits):
            key = hit.chunk.id
            chunks[key] = hit.chunk
            scores[key] = scores.get(key, 0.0) + self.keyword_weight / (self.rrf_k + rank + 1)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredChunk(chunk=chunks[key], score=score) for key, score in ranked]

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "retriever",
            "hybrid": self.hybrid,
            "store": self.store.describe().model_dump(mode="json"),
            "embedder": self.embedder.describe().model_dump(mode="json"),
        }
