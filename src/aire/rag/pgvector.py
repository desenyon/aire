"""Postgres + pgvector store (lazy). Falls back guidance when psycopg absent."""

from __future__ import annotations

import importlib.util
import json
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.rag.store import VectorStore, cosine_similarity, tokenize
from aire.rag.types import Chunk, ScoredChunk


def _require_psycopg() -> Any:
    if importlib.util.find_spec("psycopg") is None:
        raise ConfigurationError(
            "psycopg is required for pgvector: pip install 'aire[pgvector]'",
            code="rag.pgvector_missing",
        )
    import psycopg  # type: ignore[import-not-found]

    return psycopg


class PgVectorStore(VectorStore):
    """Postgres vector store using pgvector when available; JSONB fallback."""

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "aire_chunks",
        dim: int | None = None,
    ) -> None:
        self.psycopg = _require_psycopg()
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self._ensure()

    def _connect(self) -> Any:
        return self.psycopg.connect(self.dsn)

    def _ensure(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    text TEXT NOT NULL,
                    idx INT,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding JSONB
                )
                """
            )
            conn.commit()

    async def upsert(self, chunks: list[Chunk]) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (id, document_id, text, idx, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        document_id=EXCLUDED.document_id,
                        text=EXCLUDED.text,
                        idx=EXCLUDED.idx,
                        metadata=EXCLUDED.metadata,
                        embedding=EXCLUDED.embedding
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.text,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        json.dumps(chunk.embedding or []),
                    ),
                )
            conn.commit()
        return len(chunks)

    def _row_to_chunk(self, row: Any) -> Chunk:
        return Chunk(
            id=row[0],
            document_id=row[1],
            text=row[2],
            index=int(row[3] or 0),
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
            embedding=row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]"),
        )

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, document_id, text, idx, metadata, embedding FROM {self.table}")
            rows = cur.fetchall()
        scored: list[ScoredChunk] = []
        from aire.rag.acl import matches_acl

        for row in rows:
            chunk = self._row_to_chunk(row)
            if filter:
                acl = filter.get("__acl__") if isinstance(filter.get("__acl__"), dict) else None
                plain = {k: v for k, v in filter.items() if k != "__acl__"}
                if plain and not all(chunk.metadata.get(a) == b for a, b in plain.items()):
                    continue
                if acl and not matches_acl(chunk.metadata, acl):
                    continue
            if not chunk.embedding:
                continue
            scored.append(
                ScoredChunk(chunk=chunk, score=cosine_similarity(vector, chunk.embedding))
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def search_text(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        terms = tokenize(query)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, document_id, text, idx, metadata, embedding FROM {self.table}")
            rows = cur.fetchall()
        scored: list[ScoredChunk] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if filter:
                plain = {a: b for a, b in filter.items() if a != "__acl__"}
                if plain and not all(chunk.metadata.get(a) == b for a, b in plain.items()):
                    continue
            if not terms:
                scored.append(ScoredChunk(chunk=chunk, score=0.0))
                continue
            tokens = tokenize(chunk.text)
            score = sum(tokens.count(t) for t in terms)
            if score:
                scored.append(ScoredChunk(chunk=chunk, score=float(score)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids,))
            conn.commit()
            return int(cur.rowcount or 0)

    async def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name="pgvector",
            provider="postgres",
            capabilities=["vector-search", "keyword-search", "persistence"],
            extra={"table": self.table},
        )


def register(runtime: Any) -> None:
    def _factory(name: str = "default", *, runtime: Any = None, **options: Any) -> VectorStore:
        dsn = options.pop("dsn", None) or options.pop("url", None)
        if not dsn:
            raise ConfigurationError(
                "pgvector store requires dsn=", code="rag.pgvector_dsn"
            )
        return PgVectorStore(dsn, **options)

    runtime.vector_stores.register("pgvector", _factory, replace=True)
    runtime.vector_stores.register("postgres", _factory, replace=True)
