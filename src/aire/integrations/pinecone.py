"""Pinecone vector store integration (``"pinecone:<index>"``).

Speaks Pinecone's REST API over httpx — no pinecone-client dependency.
Requires the index host as ``base_url`` (or ``PINECONE_BASE_URL``) and an API
key (``PINECONE_API_KEY`` or ``providers.pinecone.api_key``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.integrations.http import ProviderHttpClient
from aire.rag.store import VectorStore, tokenize
from aire.rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


class PineconeVectorStore(VectorStore):
    """Vector store backed by a Pinecone index (serverless or pod)."""

    def __init__(self, index: str, client: ProviderHttpClient) -> None:
        self.index = index
        self.client = client

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors: list[dict[str, Any]] = []
        for c in chunks:
            metadata: dict[str, Any] = {**c.metadata, "text": c.text}
            vectors.append({"id": c.id, "values": c.embedding or [], "metadata": metadata})
        await self.client.post_json("/vectors/upsert", {"vectors": vectors})
        return len(chunks)

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        body: dict[str, Any] = {"vector": vector, "topK": k, "includeMetadata": True}
        if filter:
            body["filter"] = {key: {"$eq": value} for key, value in filter.items()}
        data = await self.client.post_json("/query", body)
        return [_to_scored(m) for m in data.get("matches", [])]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        # Pinecone has no native BM25; fetch a page and score client-side.
        data = await self.client.post_json(
            "/query",
            {
                "vector": [0.0],
                "topK": min(k * 20, 500),
                "includeMetadata": True,
            },
        )
        terms = set(tokenize(query))
        scored: list[ScoredChunk] = []
        for match in data.get("matches", []):
            hit = _to_scored(match)
            overlap = len(terms & set(tokenize(hit.chunk.text)))
            if overlap or not terms:
                scored.append(ScoredChunk(chunk=hit.chunk, score=float(overlap)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        await self.client.post_json("/vectors/delete", {"ids": ids})
        return len(ids)

    async def count(self) -> int:
        data = await self.client.post_json("/describe_index_stats", {})
        return int(data.get("totalVectorCount", 0))

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self.index,
            provider="pinecone",
            capabilities=["vector-search", "persistence"],
        )


def _to_scored(match: dict[str, Any]) -> ScoredChunk:
    metadata = dict(match.get("metadata", {}) or {})
    text = str(metadata.pop("text", ""))
    return ScoredChunk(
        chunk=Chunk(id=str(match.get("id", "")), text=text, metadata=metadata),
        score=float(match.get("score", 0.0)),
    )


def register(runtime: Runtime) -> None:
    def _factory(name: str = "aire", *, runtime: Runtime = None, **options: Any) -> VectorStore:  # type: ignore[assignment]
        cred = runtime.settings.credential("pinecone")
        import os

        base_url = options.get("base_url") or cred.base_url or os.environ.get("PINECONE_BASE_URL")
        if not base_url:
            raise ConfigurationError(
                "pinecone requires the index host: pass base_url=... or set PINECONE_BASE_URL",
                code="config.missing_base_url",
                context={"provider": "pinecone"},
            )
        api_key = options.get("api_key") or cred.resolve_key("PINECONE_API_KEY")
        headers = dict(cred.default_headers)
        if api_key:
            headers["Api-Key"] = api_key
        client = ProviderHttpClient(runtime, "pinecone", base_url=base_url, headers=headers)
        return PineconeVectorStore(name, client)

    runtime.vector_stores.register("pinecone", _factory, replace=True)
