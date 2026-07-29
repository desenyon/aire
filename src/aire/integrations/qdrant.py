"""Qdrant vector store integration (``"qdrant:<collection>"``).

Speaks to Qdrant's REST API over httpx — no qdrant-client dependency.
Activate with ``register(runtime)`` or by using ``AI.rag.vector_store("qdrant:...")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.core.errors import ProviderError
from aire.core.types import HealthStatus, Manifest
from aire.integrations.http import ProviderHttpClient
from aire.rag.filters import apply_acl_to_hits, apply_metadata_filter, split_acl_filter
from aire.rag.store import VectorStore, tokenize
from aire.rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "http://localhost:6333"


def _qdrant_must_filter(store_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a Qdrant Filter with must match conditions on ``metadata.*`` payload keys."""
    if not store_filter:
        return None
    return {
        "must": [
            {"key": f"metadata.{key}", "match": {"value": value}}
            for key, value in store_filter.items()
        ]
    }


class QdrantVectorStore(VectorStore):
    """Vector store backed by a Qdrant collection."""

    def __init__(
        self,
        collection: str,
        client: ProviderHttpClient,
        *,
        dimension: int | None = None,
    ) -> None:
        self.collection = collection
        self.client = client
        self.dimension = dimension
        self._ensured = False

    async def _ensure_collection(self, dimension: int) -> None:
        if self._ensured:
            return
        try:
            await self.client.get_json(f"/collections/{self.collection}")
        except ProviderError:
            await self.client.raw.put(
                f"/collections/{self.collection}",
                json={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
        self._ensured = True

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        dimension = len(chunks[0].embedding or [])
        await self._ensure_collection(dimension)
        points = [
            {
                "id": abs(hash(c.id)) % (2**63),
                "vector": c.embedding or [],
                "payload": {"aire_id": c.id, "text": c.text, "metadata": c.metadata},
            }
            for c in chunks
        ]
        response = await self.client.raw.put(
            f"/collections/{self.collection}/points", json={"points": points}
        )
        if response.status_code >= 400:
            raise ProviderError("qdrant", f"upsert failed: HTTP {response.status_code}")
        return len(chunks)

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        store_filter, acl = split_acl_filter(filter)
        # Over-fetch when ACL post-filter may drop hits.
        limit = k * 5 if acl else k
        body: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": True}
        native = _qdrant_must_filter(store_filter)
        if native:
            body["filter"] = native
        data = await self.client.post_json(f"/collections/{self.collection}/points/search", body)
        hits = [_to_scored(hit) for hit in data.get("result", [])]
        # If native filter failed to express a key, still enforce equality client-side.
        hits = apply_metadata_filter(hits, store_filter)
        hits = apply_acl_to_hits(hits, acl)
        return hits[:k]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        # Qdrant has no built-in BM25; scroll and score client-side for hybrid fusion.
        store_filter, acl = split_acl_filter(filter)
        body: dict[str, Any] = {"limit": min(k * 20, 500), "with_payload": True}
        native = _qdrant_must_filter(store_filter)
        if native:
            body["filter"] = native
        data = await self.client.post_json(f"/collections/{self.collection}/points/scroll", body)
        terms = set(tokenize(query))
        scored: list[ScoredChunk] = []
        for point in data.get("result", {}).get("points", []):
            hit = _to_scored(point)
            overlap = len(terms & set(tokenize(hit.chunk.text)))
            if overlap or not terms:
                scored.append(ScoredChunk(chunk=hit.chunk, score=float(overlap)))
        scored = apply_metadata_filter(scored, store_filter)
        scored = apply_acl_to_hits(scored, acl)
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        await self.client.raw.post(
            f"/collections/{self.collection}/points/delete", json={"points": ids}
        )
        return len(ids)

    async def delete_by_document(self, document_id: str) -> int:
        """Delete points whose payload metadata ``document_id`` or ``source`` matches."""
        filt = {
            "should": [
                {"key": "metadata.document_id", "match": {"value": document_id}},
                {"key": "metadata.source", "match": {"value": document_id}},
            ]
        }
        await self.client.raw.post(
            f"/collections/{self.collection}/points/delete",
            json={"filter": filt},
        )
        # Qdrant filter-delete does not return a count; best-effort.
        return 0

    async def count(self) -> int:
        data = await self.client.get_json(f"/collections/{self.collection}")
        return int(data.get("result", {}).get("points_count", 0))

    async def health(self) -> HealthStatus:
        try:
            await self.client.raw.get("/healthz")
        except Exception as exc:
            return HealthStatus.unhealthy(f"qdrant unreachable: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self.collection,
            provider="qdrant",
            capabilities=["vector-search", "persistence"],
        )


def _to_scored(hit: dict[str, Any]) -> ScoredChunk:
    payload = hit.get("payload", {}) or {}
    chunk = Chunk(
        id=str(payload.get("aire_id", hit.get("id", ""))),
        text=str(payload.get("text", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )
    return ScoredChunk(chunk=chunk, score=float(hit.get("score", 0.0)))


def register(runtime: Runtime) -> None:
    def _factory(name: str = "aire", *, runtime: Runtime = None, **options: Any) -> VectorStore:  # type: ignore[assignment]
        cred = runtime.settings.credential("qdrant")
        base_url = options.get("base_url") or cred.base_url or DEFAULT_BASE_URL
        headers = dict(cred.default_headers)
        api_key = options.get("api_key") or cred.resolve_key("QDRANT_API_KEY")
        if api_key:
            headers["api-key"] = api_key
        client = ProviderHttpClient(runtime, "qdrant", base_url=base_url, headers=headers)
        return QdrantVectorStore(name, client)

    runtime.vector_stores.register("qdrant", _factory, replace=True)
