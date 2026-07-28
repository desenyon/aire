"""Embedded SQLite vector store (``sqlite:<path>``).

Single-file, transactional, zero dependencies (stdlib ``sqlite3``) — the
aire-native embedded store. Search semantics are identical to
:class:`~aire.rag.store.LocalVectorStore` (same BM25 + cosine implementation);
this adds durable, incremental persistence: every upsert/delete is written
through, so gigabyte-scale JSON snapshots are never rewritten.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from aire.core.types import Manifest
from aire.rag.store import LocalVectorStore
from aire.rag.types import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding TEXT
);
"""


class SQLiteVectorStore(LocalVectorStore):
    """LocalVectorStore semantics with SQLite write-through persistence."""

    def __init__(self, path: str | Path, *, name: str = "sqlite") -> None:
        self._sqlite_path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        super().__init__(path=None, name=name)
        self._load_sqlite()

    def _load_sqlite(self) -> None:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, document_id, idx, text, metadata, embedding FROM chunks"
            ).fetchall()
        for row in rows:
            self._chunks[row[0]] = Chunk(
                id=row[0],
                document_id=row[1],
                index=row[2],
                text=row[3],
                metadata=json.loads(row[4]),
                embedding=json.loads(row[5]) if row[5] else None,
            )

    async def upsert(self, chunks: list[Chunk]) -> int:
        count = await super().upsert(chunks)
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO chunks"
                "(id, document_id, idx, text, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.document_id,
                        c.index,
                        c.text,
                        json.dumps(c.metadata),
                        json.dumps(c.embedding) if c.embedding is not None else None,
                    )
                    for c in chunks
                ],
            )
        return count

    async def delete(self, ids: list[str]) -> int:
        removed = await super().delete(ids)
        if removed:
            with self._lock, self._db:
                self._db.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])
        return removed

    async def clear(self) -> None:
        await super().clear()
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks")

    async def aclose(self) -> None:
        import asyncio

        await asyncio.to_thread(self._db.close)

    def describe(self) -> Manifest:
        return Manifest(
            kind="vector_store",
            name=self._name,
            provider="sqlite",
            capabilities=["vector-search", "keyword-search", "embedded-persistence"],
            extra={"count": len(self._chunks), "path": self._sqlite_path},
        )


def register(runtime: Any) -> None:
    """Register the embedded SQLite vector store factory on a runtime."""

    def _factory(
        name: str = "vectors", *, runtime: Any = None, **options: Any
    ) -> SQLiteVectorStore:
        path = options.pop("path", None) or name
        if path in {"memory", ":memory:"}:
            path = ":memory:"
        return SQLiteVectorStore(path, **options)

    runtime.vector_stores.register("sqlite", _factory, replace=True)
