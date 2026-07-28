"""RAG primitives: documents, chunks, retrieval results, citations, answers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import Usage, new_id


class Document(BaseModel):
    """A source document prior to chunking."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str | None:
        value = self.metadata.get("source")
        return str(value) if value is not None else None


class Chunk(BaseModel):
    """A retrievable unit of text with optional embedding."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str = ""
    text: str
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, exclude=True)


class ScoredChunk(BaseModel):
    """A chunk paired with a relevance score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


class Citation(BaseModel):
    """Verifiable pointer from an answer back to its source."""

    source: str
    chunk_id: str
    excerpt: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """A grounded answer with citations and usage accounting."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = "unknown"
    retrieved: int = 0


class IndexReport(BaseModel):
    """Outcome of an ingestion run."""

    documents: int
    chunks: int
    store: str
    embedder: str
    duration_ms: float = 0.0
