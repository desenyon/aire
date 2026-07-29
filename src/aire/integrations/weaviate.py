"""Weaviate vector store integration (``"weaviate:<class>"``).

Speaks Weaviate's REST + GraphQL APIs over httpx — no weaviate-client
dependency. Weaviate has native BM25, so ``search_text`` runs server-side.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aire.core.types import HealthStatus, Manifest
from aire.integrations.http import ProviderHttpClient
from aire.rag.filters import apply_acl_to_hits, apply_metadata_filter, split_acl_filter
from aire.rag.store import VectorStore
from aire.rag.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "http://localhost:8080"


class WeaviateVectorStore(VectorStore):
    """Vector store backed by a Weaviate class (native vector + BM25 search)."""

    def __init__(self, collection: str, client: ProviderHttpClient) -> None:
        self.collection = collection
        self.client = client
        self._class = collection[:1].upper() + collection[1:]
        self._ensured = False

    async def _ensure_class(self) -> None:
        if self._ensured:
            return
        response = await self.client.raw.get(f"/v1/schema/{self._class}")
        if response.status_code == 404:
            await self.client.raw.post(
                "/v1/schema",
                json={
                    "class": self._class,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "aire_id", "dataType": ["text"]},
                        {"name": "text", "dataType": ["text"]},
                        {"name": "metadata_json", "dataType": ["text"]},
                    ],
                },
            )
        self._ensured = True

    async def upsert(self, chunks: list[Chunk]) -> int:
        await self._ensure_class()
        objects = [
            {
                "class": self._class,
                "vector": c.embedding or [],
                "properties": {
                    "aire_id": c.id,
                    "text": c.text,
                    "metadata_json": json.dumps(c.metadata),
                },
            }
            for c in chunks
        ]
        await self.client.post_json("/v1/batch/objects", {"objects": objects})
        return len(chunks)

    async def _graphql(self, query: str) -> Any:
        data = await self.client.post_json("/v1/graphql", {"query": query})
        return data.get("data", {}).get("Get", {}).get(self._class, [])

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        await self._ensure_class()
        store_filter, acl = split_acl_filter(filter)
        # Metadata lives in a JSON text property — GraphQL where on nested keys is
        # awkward; over-fetch then apply equality + ACL client-side.
        limit = k * 5 if (store_filter or acl) else k
        query = (
            "{ Get { "
            f"{self._class}(nearVector: {{vector: {json.dumps(vector)}}}, limit: {limit})"
            " { aire_id text metadata_json _additional { distance id } } } }"
        )
        rows = await self._graphql(query)
        hits = [_to_scored(row) for row in rows] if isinstance(rows, list) else []
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
        await self._ensure_class()
        store_filter, acl = split_acl_filter(filter)
        limit = k * 5 if (store_filter or acl) else k
        safe = json.dumps(query)
        gql = (
            "{ Get { "
            f'{self._class}(bm25: {{query: {safe}, properties: ["text"]}}, limit: {limit})'
            " { aire_id text metadata_json _additional { score id } } } }"
        )
        rows = await self._graphql(gql)
        if not isinstance(rows, list):
            return []
        hits = [_to_scored(row, score_key="score") for row in rows]
        hits = apply_metadata_filter(hits, store_filter)
        hits = apply_acl_to_hits(hits, acl)
        return hits[:k]

    async def delete(self, ids: list[str]) -> int:
        await self._ensure_class()
        removed = 0
        for chunk_id in ids:
            response = await self.client.raw.delete(f"/v1/objects/{self._class}/{chunk_id}")
            removed += int(response.status_code < 400)
        return removed

    async def delete_by_document(self, document_id: str) -> int:
        """Best-effort: page objects, match metadata document_id/source, delete by aire_id."""
        await self._ensure_class()
        gql = (
            "{ Get { "
            f"{self._class}(limit: 10000)"
            " { aire_id text metadata_json _additional { id } } } }"
        )
        rows = await self._graphql(gql)
        if not isinstance(rows, list):
            return 0
        ids: list[str] = []
        for row in rows:
            hit = _to_scored(row)
            meta = hit.chunk.metadata
            if meta.get("document_id") == document_id or meta.get("source") == document_id:
                ids.append(hit.chunk.id)
        if not ids:
            return 0
        return await self.delete(ids)

    async def count(self) -> int:
        await self._ensure_class()
        data = await self.client.post_json(
            "/v1/graphql",
            {"query": "{ Aggregate { " + self._class + " { meta { count } } } }"},
        )
        rows = data.get("data", {}).get("Aggregate", {}).get(self._class, [])
        return int(rows[0]["meta"]["count"]) if rows else 0

    async def health(self) -> HealthStatus:
        try:
            await self.client.raw.get("/v1/.well-known/ready")
        except Exception as exc:
            return HealthStatus.unhealthy(f"weaviate unreachable: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self.collection,
            provider="weaviate",
            capabilities=["vector-search", "keyword-search", "persistence"],
        )


def _to_scored(row: dict[str, Any], *, score_key: str = "distance") -> ScoredChunk:
    additional = row.get("_additional", {}) or {}
    raw = float(additional.get(score_key, 0.0) or 0.0)
    score = 1.0 - min(raw, 1.0) if score_key == "distance" else raw
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return ScoredChunk(
        chunk=Chunk(
            id=str(row.get("aire_id", "")),
            text=str(row.get("text", "")),
            metadata=metadata,
            document_id=str(metadata.get("document_id") or ""),
        ),
        score=score,
    )


def register(runtime: Runtime) -> None:
    def _factory(name: str = "aire", *, runtime: Runtime = None, **options: Any) -> VectorStore:  # type: ignore[assignment]
        cred = runtime.settings.credential("weaviate")
        base_url = options.get("base_url") or cred.base_url or DEFAULT_BASE_URL
        api_key = options.get("api_key") or cred.resolve_key("WEAVIATE_API_KEY")
        headers = dict(cred.default_headers)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client = ProviderHttpClient(runtime, "weaviate", base_url=base_url, headers=headers)
        return WeaviateVectorStore(name, client)

    runtime.vector_stores.register("weaviate", _factory, replace=True)
