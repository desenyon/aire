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
from aire.integrations.vector_codec import pack_metadata, unpack_chunk
from aire.rag.filters import apply_acl_to_hits, apply_metadata_filter, split_acl_filter
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
            metadata: dict[str, Any] = {**pack_metadata(c), "text": c.text}
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
        if k <= 0:
            return []
        store_filter, acl = split_acl_filter(filter)
        limit = max(k, await self.count()) if acl else k
        body: dict[str, Any] = {"vector": vector, "topK": limit, "includeMetadata": True}
        if store_filter:
            body["filter"] = {key: {"$eq": value} for key, value in store_filter.items()}
        data = await self.client.post_json("/query", body)
        hits = [_to_scored(m) for m in data.get("matches", [])]
        return apply_acl_to_hits(apply_metadata_filter(hits, store_filter), acl)[:k]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        matches: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": 100}
        seen_tokens = set()
        while True:
            response = await self.client.request("GET", "/vectors/list", params=params)
            page = self.client.parse_json(response)
            ids = [v["id"] for v in page.get("vectors", [])]
            if ids:
                response = await self.client.request(
                    "GET", "/vectors/fetch", params=[("ids", i) for i in ids]
                )
                values = self.client.parse_json(response).get("vectors", {})
                matches.extend(values.values())
            token = page.get("pagination", {}).get("next")
            if not token:
                break
            if token in seen_tokens:
                from aire.core.errors import ProviderError

                raise ProviderError("pinecone", "repeated listing token", retryable=False)
            seen_tokens.add(token)
            params["paginationToken"] = token
        terms = set(tokenize(query))
        scored: list[ScoredChunk] = []
        from aire.rag.acl import matches_acl

        for match in matches:
            hit = _to_scored(match)
            if filter:
                acl = filter.get("__acl__") if isinstance(filter.get("__acl__"), dict) else None
                if acl and not matches_acl(hit.chunk.metadata, acl):
                    continue
            overlap = len(terms & set(tokenize(hit.chunk.text)))
            if overlap or not terms:
                scored.append(ScoredChunk(chunk=hit.chunk, score=float(overlap)))
        store_filter, acl = split_acl_filter(filter)
        scored = apply_acl_to_hits(apply_metadata_filter(scored, store_filter), acl)
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
        chunk=unpack_chunk(str(match.get("id", "")), text, metadata),
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
