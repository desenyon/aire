"""Postgres vector store with optional pgvector; honest JSONB fallback."""

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
    """Postgres-backed vector store.

    When the ``vector`` extension is available and ``dim`` is set, embeddings
    use a real ``vector(dim)`` column with ``<=>`` distance. Otherwise embeddings
    are stored as JSONB and scored in Python (advertised as ``jsonb_fallback``).
    Keyword search uses Postgres full-text search when available.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "aire_chunks",
        dim: int | None = None,
        use_pgvector: bool | None = None,
    ) -> None:
        self.psycopg = _require_psycopg()
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self._pgvector = False
        self._fts = False
        self._use_pgvector_pref = use_pgvector
        self._ensure()

    def _connect(self) -> Any:
        return self.psycopg.connect(self.dsn)

    def _ensure(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            want = self._use_pgvector_pref
            if want is not False and self.dim is not None:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                    self._pgvector = True
                except Exception:
                    conn.rollback()
                    self._pgvector = False
            if self._pgvector and self.dim is not None:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        id TEXT PRIMARY KEY,
                        document_id TEXT,
                        text TEXT NOT NULL,
                        idx INT,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding vector({int(self.dim)})
                    )
                    """
                )
            else:
                self._pgvector = False
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
            try:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table}_fts
                    ON {self.table} USING gin (to_tsvector('english', text))
                    """
                )
                self._fts = True
            except Exception:
                conn.rollback()
                self._fts = False
            conn.commit()

    async def upsert(self, chunks: list[Chunk]) -> int:
        if self._pgvector and self.dim is None and chunks and chunks[0].embedding:
            # Late dim discovery cannot ALTER a vector(N) column safely in-place.
            raise ConfigurationError(
                "pgvector store was created without dim=; pass dim=<embedding size> "
                "at construction (recreating the table is required to switch to ANN). "
                "Use use_pgvector=False for JSONB + Python cosine fallback.",
                code="rag.pgvector_late_dim",
                context={"table": self.table, "embedding_len": len(chunks[0].embedding or [])},
            )
        with self._connect() as conn, conn.cursor() as cur:
            for chunk in chunks:
                emb = chunk.embedding or []
                if self._pgvector:
                    if self.dim is not None and len(emb) != self.dim:
                        raise ConfigurationError(
                            f"embedding dim {len(emb)} != store dim {self.dim}",
                            code="rag.pgvector_dim",
                        )
                    cur.execute(
                        f"""
                        INSERT INTO {self.table} (id, document_id, text, idx, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
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
                            emb,
                        ),
                    )
                else:
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
                            json.dumps(emb),
                        ),
                    )
            conn.commit()
        return len(chunks)

    def _row_to_chunk(self, row: Any) -> Chunk:
        emb = row[5]
        if isinstance(emb, str):
            emb = json.loads(emb)
        elif emb is not None and not isinstance(emb, list):
            emb = list(emb)
        return Chunk(
            id=row[0],
            document_id=row[1],
            text=row[2],
            index=int(row[3] or 0),
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
            embedding=emb or [],
        )

    def _filter_ok(self, chunk: Chunk, filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        from aire.rag.acl import matches_acl

        acl = filter.get("__acl__") if isinstance(filter.get("__acl__"), dict) else None
        plain = {k: v for k, v in filter.items() if k != "__acl__"}
        if plain and not all(chunk.metadata.get(a) == b for a, b in plain.items()):
            return False
        return not (acl and not matches_acl(chunk.metadata, acl))

    async def search(
        self,
        vector: list[float],
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if self._pgvector:
            return await self._search_pgvector(vector, k=k, filter=filter)
        return await self._search_jsonb(vector, k=k, filter=filter)

    async def _search_pgvector(
        self,
        vector: list[float],
        *,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[ScoredChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, document_id, text, idx, metadata, embedding,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, vector, max(k * 5, k)),
            )
            rows = cur.fetchall()
        scored: list[ScoredChunk] = []
        for row in rows:
            chunk = self._row_to_chunk(row[:6])
            if not self._filter_ok(chunk, filter):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=float(row[6] or 0.0)))
            if len(scored) >= k:
                break
        return scored

    async def _search_jsonb(
        self,
        vector: list[float],
        *,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[ScoredChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, document_id, text, idx, metadata, embedding FROM {self.table}")
            rows = cur.fetchall()
        scored: list[ScoredChunk] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if not self._filter_ok(chunk, filter):
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
        if self._fts:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, document_id, text, idx, metadata, embedding,
                           ts_rank(to_tsvector('english', text), plainto_tsquery('english', %s))
                    FROM {self.table}
                    WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
                    ORDER BY 7 DESC
                    LIMIT %s
                    """,
                    (query, query, max(k * 5, k)),
                )
                rows = cur.fetchall()
            scored: list[ScoredChunk] = []
            for row in rows:
                chunk = self._row_to_chunk(row[:6])
                if not self._filter_ok(chunk, filter):
                    continue
                scored.append(ScoredChunk(chunk=chunk, score=float(row[6] or 0.0)))
                if len(scored) >= k:
                    break
            if scored:
                return scored
        # Fallback: client-side token overlap (not advertised as keyword-search)
        terms = tokenize(query)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, document_id, text, idx, metadata, embedding FROM {self.table}")
            rows = cur.fetchall()
        scored = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if not self._filter_ok(chunk, filter):
                continue
            if not terms:
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
        caps = ["vector-search", "persistence"]
        if self._fts:
            caps.append("keyword-search")
        backend = "pgvector" if self._pgvector else "jsonb_fallback"
        return Manifest(
            kind="vector_store",
            name="pgvector",
            provider="postgres",
            capabilities=caps,
            extra={
                "table": self.table,
                "backend": backend,
                "dim": self.dim,
                "fts": self._fts,
                "note": (
                    "Pass dim= and ensure CREATE EXTENSION vector for ANN; "
                    "otherwise embeddings use JSONB + Python cosine."
                ),
            },
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
