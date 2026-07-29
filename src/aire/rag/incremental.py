"""Incremental index helpers over a Knowledge pipeline / VectorStore."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from aire.core.errors import ConfigurationError
from aire.rag.pipeline import Knowledge
from aire.rag.types import Document, IndexReport


class IncrementalIndex:
    """Document-level incremental updates for a :class:`~aire.rag.pipeline.Knowledge`."""

    def __init__(self, knowledge: Knowledge) -> None:
        self.knowledge = knowledge

    async def add_documents(
        self,
        documents: Sequence[Document | str | dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> IndexReport:
        docs = _coerce_documents(documents)
        return await self.knowledge.ingest(docs, metadata=metadata)

    async def update_document(self, document: Document | str | dict[str, Any]) -> IndexReport:
        """Replace chunks for one document id (delete-then-ingest)."""
        doc = _coerce_documents([document])[0]
        started = time.perf_counter()
        chunks = await self.knowledge.reindex_document(doc)
        return IndexReport(
            documents=1,
            chunks=chunks,
            store=type(self.knowledge.store).__name__,
            embedder=(await self.knowledge._embedder()).name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def delete_by_document(self, document_id: str) -> int:
        store = self.knowledge.store
        if hasattr(store, "delete_by_document"):
            return int(await store.delete_by_document(document_id))
        # Fallback: page via search_text, match document_id/source, delete by ids.
        try:
            hits = await store.search_text("", k=10_000)
        except Exception as exc:
            raise ConfigurationError(
                f"{type(store).__name__} does not support delete_by_document "
                f"and search_text fallback failed: {exc}",
                code="rag.incremental_delete",
                context={"store": type(store).__name__},
            ) from exc
        ids: list[str] = []
        for hit in hits:
            meta = hit.chunk.metadata or {}
            if (
                hit.chunk.document_id == document_id
                or meta.get("document_id") == document_id
                or meta.get("source") == document_id
            ):
                ids.append(hit.chunk.id)
        if not ids:
            return 0
        return int(await store.delete(ids))

    async def delete_ids(self, ids: list[str]) -> int:
        return int(await self.knowledge.store.delete(ids))

    async def reindex(
        self,
        source: Any,
        *,
        clear: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> IndexReport:
        if clear and hasattr(self.knowledge.store, "clear"):
            await self.knowledge.store.clear()
        return await self.knowledge.ingest(source, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        store = self.knowledge.store
        return {
            "kind": "incremental_index",
            "store": type(store).__name__,
            "supports_delete_by_document": hasattr(store, "delete_by_document"),
            "supports_clear": hasattr(store, "clear"),
            "knowledge_update": hasattr(self.knowledge, "reindex_document"),
        }


def wrap(knowledge: Knowledge) -> IncrementalIndex:
    return IncrementalIndex(knowledge)


def _coerce_documents(
    documents: Sequence[Document | str | dict[str, Any]],
) -> list[Document]:
    out: list[Document] = []
    for item in documents:
        if isinstance(item, Document):
            out.append(item)
        elif isinstance(item, str):
            out.append(Document(text=item))
        elif isinstance(item, dict):
            out.append(Document.model_validate(item))
        else:
            raise ConfigurationError(
                f"unsupported document type {type(item).__name__}",
                code="rag.incremental_doc",
            )
    return out
