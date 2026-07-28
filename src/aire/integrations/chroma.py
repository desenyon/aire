"""Chroma vector store integration (``"chroma:<collection>"``).

Uses Chroma's HTTP API over httpx — no chromadb dependency required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.core.errors import ProviderError
from aire.core.types import HealthStatus, Manifest
from aire.integrations.http import ProviderHttpClient
from aire.rag.store import VectorStore, tokenize
from aire.rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "http://localhost:8000"


class ChromaVectorStore(VectorStore):
    """Vector store backed by a Chroma collection (v2 HTTP API)."""

    def __init__(self, collection: str, client: ProviderHttpClient) -> None:
        self.collection = collection
        self.client = client
        self._collection_id: str | None = None

    async def _id(self) -> str:
        if self._collection_id is None:
            data = await self.client.post_json(
                "/api/v2/tenants/default_tenant/databases/default_database/collections",
                {"name": self.collection, "get_or_create": True},
            )
            self._collection_id = str(data.get("id"))
            if not self._collection_id or self._collection_id == "None":
                raise ProviderError("chroma", "failed to resolve collection id", retryable=False)
        return self._collection_id

    async def upsert(self, chunks: list[Chunk]) -> int:
        cid = await self._id()
        await self.client.raw.post(
            f"/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/upsert",
            json={
                "ids": [c.id for c in chunks],
                "embeddings": [c.embedding or [] for c in chunks],
                "documents": [c.text for c in chunks],
                "metadatas": [c.metadata for c in chunks],
            },
        )
        return len(chunks)

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        cid = await self._id()
        body: dict[str, Any] = {"query_embeddings": [vector], "n_results": k}
        if filter:
            body["where"] = filter
        data = await self.client.post_json(
            f"/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/query",
            body,
        )
        ids = (data.get("ids") or [[]])[0]
        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]
        results = []
        for i, chunk_id in enumerate(ids):
            chunk = Chunk(
                id=str(chunk_id),
                text=str(docs[i]) if i < len(docs) else "",
                metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
            )
            distance = float(distances[i]) if i < len(distances) else 1.0
            results.append(ScoredChunk(chunk=chunk, score=1.0 - min(distance, 1.0)))
        return results

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        cid = await self._id()
        data = await self.client.post_json(
            f"/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/get",
            {"limit": min(k * 20, 500)},
        )
        terms = set(tokenize(query))
        scored: list[ScoredChunk] = []
        for i, chunk_id in enumerate(data.get("ids", [])):
            text = str((data.get("documents") or [""])[i])
            overlap = len(terms & set(tokenize(text)))
            if overlap or not terms:
                metas = data.get("metadatas") or []
                chunk = Chunk(
                    id=str(chunk_id),
                    text=text,
                    metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
                )
                scored.append(ScoredChunk(chunk=chunk, score=float(overlap)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        cid = await self._id()
        await self.client.raw.post(
            f"/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/delete",
            json={"ids": ids},
        )
        return len(ids)

    async def count(self) -> int:
        cid = await self._id()
        # Chroma's count endpoint returns a bare JSON integer, not an object.
        response = await self.client.raw.get(
            f"/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/count"
        )
        data = response.json()
        return int(data) if isinstance(data, int) else int(data.get("count", 0))

    async def health(self) -> HealthStatus:
        try:
            await self.client.raw.get("/api/v2/heartbeat")
        except Exception as exc:
            return HealthStatus.unhealthy(f"chroma unreachable: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self.collection,
            provider="chroma",
            capabilities=["vector-search", "persistence"],
        )


def register(runtime: Runtime) -> None:
    def _factory(name: str = "aire", *, runtime: Runtime = None, **options: Any) -> VectorStore:  # type: ignore[assignment]
        cred = runtime.settings.credential("chroma")
        base_url = options.get("base_url") or cred.base_url or DEFAULT_BASE_URL
        client = ProviderHttpClient(
            runtime, "chroma", base_url=base_url, headers=dict(cred.default_headers)
        )
        return ChromaVectorStore(name, client)

    runtime.vector_stores.register("chroma", _factory, replace=True)
