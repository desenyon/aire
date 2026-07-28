"""Milvus vector store integration (``"milvus:<collection>"``).

Speaks the Milvus v2 RESTful API (Milvus 2.4+) over httpx — no pymilvus
dependency. Auth via ``MILVUS_TOKEN`` (default ``root:Milvus`` for local).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aire.core.types import Manifest
from aire.integrations.http import ProviderHttpClient
from aire.rag.store import VectorStore, tokenize
from aire.rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "http://localhost:19530"


class MilvusVectorStore(VectorStore):
    """Vector store backed by a Milvus collection (v2 REST API)."""

    def __init__(self, collection: str, client: ProviderHttpClient) -> None:
        self.collection = collection
        self.client = client
        self._ensured = False

    async def _ensure_collection(self, dimension: int) -> None:
        if self._ensured:
            return
        response = await self.client.raw.post(
            "/v2/vectordb/collections/has", json={"collectionName": self.collection}
        )
        has = response.json().get("data", {}).get("has", False)
        if not has:
            await self.client.post_json(
                "/v2/vectordb/collections/create",
                {"collectionName": self.collection, "dimension": dimension},
            )
        self._ensured = True

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        await self._ensure_collection(len(chunks[0].embedding or []))
        data = [
            {
                "id": c.id,
                "vector": c.embedding or [],
                "text": c.text,
                "metadata": json.dumps(c.metadata),
            }
            for c in chunks
        ]
        await self.client.post_json(
            "/v2/vectordb/entities/upsert", {"collectionName": self.collection, "data": data}
        )
        return len(chunks)

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        body: dict[str, Any] = {
            "collectionName": self.collection,
            "data": [vector],
            "limit": k,
            "outputFields": ["id", "text", "metadata"],
        }
        data = await self.client.post_json("/v2/vectordb/entities/search", body)
        return [_to_scored(row) for row in data.get("data", [])]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        # Milvus vector search only; page rows and score client-side for hybrid fusion.
        data = await self.client.post_json(
            "/v2/vectordb/entities/query",
            {
                "collectionName": self.collection,
                "filter": "id != ''",
                "limit": min(k * 20, 500),
                "outputFields": ["id", "text", "metadata"],
            },
        )
        terms = set(tokenize(query))
        scored: list[ScoredChunk] = []
        for row in data.get("data", []):
            hit = _to_scored(row)
            overlap = len(terms & set(tokenize(hit.chunk.text)))
            if overlap or not terms:
                scored.append(ScoredChunk(chunk=hit.chunk, score=float(overlap)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        quoted = ", ".join(json.dumps(i) for i in ids)
        await self.client.post_json(
            "/v2/vectordb/entities/delete",
            {"collectionName": self.collection, "filter": f"id in [{quoted}]"},
        )
        return len(ids)

    async def count(self) -> int:
        data = await self.client.post_json(
            "/v2/vectordb/collections/get_stats", {"collectionName": self.collection}
        )
        return int(data.get("data", {}).get("rowCount", 0))

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self.collection,
            provider="milvus",
            capabilities=["vector-search", "persistence"],
        )


def _to_scored(row: dict[str, Any]) -> ScoredChunk:
    try:
        metadata = json.loads(row.get("metadata") or "{}")
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    distance = float(row.get("distance", 0.0) or 0.0)
    return ScoredChunk(
        chunk=Chunk(id=str(row.get("id", "")), text=str(row.get("text", "")), metadata=metadata),
        score=distance,
    )


def register(runtime: Runtime) -> None:
    def _factory(name: str = "aire", *, runtime: Runtime = None, **options: Any) -> VectorStore:  # type: ignore[assignment]
        cred = runtime.settings.credential("milvus")
        base_url = options.get("base_url") or cred.base_url or DEFAULT_BASE_URL
        token = options.get("api_key") or cred.resolve_key("MILVUS_TOKEN") or "root:Milvus"
        headers = dict(cred.default_headers)
        headers["Authorization"] = f"Bearer {token}"
        client = ProviderHttpClient(runtime, "milvus", base_url=base_url, headers=headers)
        return MilvusVectorStore(name, client)

    runtime.vector_stores.register("milvus", _factory, replace=True)
