"""Text chunkers used by RAG ingestion and context packing."""

from __future__ import annotations

import re
import warnings
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel


class TextChunk(BaseModel):
    """A slice of a source text with character offsets."""

    text: str
    start: int
    end: int
    index: int


@runtime_checkable
class Chunker(Protocol):
    """Splits text into chunks."""

    def chunk(self, text: str) -> list[TextChunk]: ...


class FixedChunker:
    """Fixed-size character windows with overlap."""

    def __init__(self, size: int = 800, overlap: int = 100) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        step = self.size - self.overlap
        index = 0
        for start in range(0, len(text), step):
            end = min(start + self.size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(TextChunk(text=piece, start=start, end=end, index=index))
                index += 1
            if end >= len(text):
                break
        return chunks

    def describe(self) -> dict[str, Any]:
        return {"kind": "chunker", "name": "fixed", "size": self.size, "overlap": self.overlap}


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n{2,}")


class SentenceChunker:
    """Sentence-aware chunker that packs sentences up to ``size`` characters."""

    def __init__(self, size: int = 800, overlap_sentences: int = 1) -> None:
        self.size = size
        self.overlap_sentences = overlap_sentences
        self._label = "sentence"

    def chunk(self, text: str) -> list[TextChunk]:
        sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
        chunks: list[TextChunk] = []
        current: list[str] = []
        current_start = 0
        cursor = 0
        index = 0

        def _flush(end: int) -> None:
            nonlocal index
            piece = " ".join(current).strip()
            if piece:
                chunks.append(TextChunk(text=piece, start=current_start, end=end, index=index))
                index += 1

        for sentence in sentences:
            position = text.find(sentence, cursor)
            if position == -1:
                position = cursor
            candidate_len = sum(len(s) for s in current) + len(sentence) + len(current)
            if current and candidate_len > self.size:
                _flush(position)
                overlap = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current = [*overlap, sentence]
                current_start = position - sum(len(s) + 1 for s in overlap)
            else:
                if not current:
                    current_start = position
                current.append(sentence)
            cursor = position + len(sentence)
        _flush(len(text))
        return chunks

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "chunker",
            "name": self._label,
            "size": self.size,
        }
        if self._label != "sentence":
            out["note"] = "sentence-boundary packing; not embedding-aware"
        return out


class EmbeddingSemanticChunker:
    """Groups sentences by embedding cosine similarity when an embedder is provided.

    Without ``embedder``, falls back to sentence-boundary packing (same as
    :class:`SentenceChunker`) — a lexical/structural approximation of semantic
    chunking, not true embedding-aware segmentation. Pass ``embedder=`` to
    ``get_chunker("semantic", embedder=...)`` for cosine-based merging.
    """

    def __init__(
        self,
        size: int = 800,
        overlap_sentences: int = 1,
        *,
        embedder: Any | None = None,
        threshold: float = 0.55,
    ) -> None:
        self.size = size
        self.overlap_sentences = overlap_sentences
        self.embedder = embedder
        self.threshold = threshold

    def chunk(self, text: str) -> list[TextChunk]:
        if self.embedder is None:
            # Honest fallback: sentence packing, not embedding semantics.
            return SentenceChunker(self.size, self.overlap_sentences).chunk(text)
        return self._chunk_with_embeddings(text)

    def _chunk_with_embeddings(self, text: str) -> list[TextChunk]:
        from aire.models.base import run_sync
        from aire.rag.store import cosine_similarity

        sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return []
        if self.embedder is None:
            return []
        vectors = run_sync(self.embedder.embed_texts(sentences))
        chunks: list[TextChunk] = []
        group: list[str] = []
        group_start = 0
        cursor = 0
        index = 0
        prev_vec: list[float] | None = None

        def _flush(end: int) -> None:
            nonlocal index, group, prev_vec
            piece = " ".join(group).strip()
            if piece:
                chunks.append(TextChunk(text=piece, start=group_start, end=end, index=index))
                index += 1
            group = []
            prev_vec = None

        for sentence, vec in zip(sentences, vectors, strict=True):
            position = text.find(sentence, cursor)
            if position == -1:
                position = cursor
            candidate_len = sum(len(s) for s in group) + len(sentence) + len(group)
            sim = cosine_similarity(prev_vec, vec) if prev_vec is not None else 1.0
            split = bool(group) and (candidate_len > self.size or sim < self.threshold)
            if split:
                _flush(position)
            if not group:
                group_start = position
            group.append(sentence)
            prev_vec = vec
            cursor = position + len(sentence)
        _flush(len(text))
        return chunks

    def describe(self) -> dict[str, Any]:
        if self.embedder is None:
            return {
                "kind": "chunker",
                "name": "semantic",
                "mode": "sentence_approximation",
                "size": self.size,
                "note": "no embedder; sentence-boundary packing only",
            }
        return {
            "kind": "chunker",
            "name": "semantic",
            "mode": "embedding",
            "size": self.size,
            "threshold": self.threshold,
        }


# Alias kept for catalog honesty: semantic_sentence == SentenceChunker.
class SemanticSentenceChunker(SentenceChunker):
    """Explicit sentence-based semantic approximation (no embeddings)."""

    def __init__(self, size: int = 800, overlap_sentences: int = 1) -> None:
        super().__init__(size=size, overlap_sentences=overlap_sentences)
        self._label = "semantic_sentence"


class RecursiveChunker:
    """Split on a separator hierarchy (paragraphs → sentences → words → chars)."""

    def __init__(self, size: int = 800, separators: list[str] | None = None) -> None:
        self.size = size
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[TextChunk]:
        pieces = self._split(text, self.separators)
        chunks: list[TextChunk] = []
        cursor = 0
        for i, piece in enumerate(pieces):
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            chunks.append(TextChunk(text=piece, start=start, end=start + len(piece), index=i))
            cursor = start + len(piece)
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.size:
            return [text.strip()] if text.strip() else []
        separator = separators[0]
        rest = separators[1:] or [""]
        if separator == "":
            return [text[i : i + self.size].strip() for i in range(0, len(text), self.size)]
        segments = text.split(separator)
        merged: list[str] = []
        current = ""
        for segment in segments:
            candidate = f"{current}{separator}{segment}" if current else segment
            if len(candidate) <= self.size:
                current = candidate
            else:
                if current:
                    merged.extend(
                        self._split(current, rest) if len(current) > self.size else [current]
                    )
                current = segment
        if current:
            merged.extend(self._split(current, rest) if len(current) > self.size else [current])
        return [m.strip() for m in merged if m.strip()]

    def describe(self) -> dict[str, Any]:
        return {"kind": "chunker", "name": "recursive", "size": self.size}


_CHUNKERS: dict[str, type] = {
    "fixed": FixedChunker,
    "sentence": SentenceChunker,
    "semantic_sentence": SemanticSentenceChunker,
    "semantic": EmbeddingSemanticChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(name: str = "recursive", **options: object) -> Chunker:
    """Resolve a chunker by name.

    ``semantic`` uses :class:`EmbeddingSemanticChunker`. Pass ``embedder=`` for
    cosine-based sentence grouping; without an embedder it is a sentence-boundary
    approximation (see ``describe()``).
    """
    if name == "semantic" and "embedder" not in options:
        # Keep working without embedder, but make the approximation visible.
        warnings.warn(
            "get_chunker('semantic') without embedder uses sentence-boundary packing "
            "(not embedding similarity). Pass embedder=... or use 'semantic_sentence'.",
            UserWarning,
            stacklevel=2,
        )
    try:
        cls = _CHUNKERS[name]
    except KeyError:
        from aire.core.errors import NotFoundError

        raise NotFoundError("chunker", name, context={"available": sorted(_CHUNKERS)}) from None
    # EmbeddingSemanticChunker accepts embedder=; SentenceChunker does not — strip unknown.
    if cls is EmbeddingSemanticChunker:
        return EmbeddingSemanticChunker(**options)  # type: ignore[arg-type]
    clean = {
        k: v
        for k, v in options.items()
        if k in ("size", "overlap", "overlap_sentences", "separators")
    }
    if cls is FixedChunker:
        clean = {k: v for k, v in options.items() if k in ("size", "overlap")}
    elif cls in (SentenceChunker, SemanticSentenceChunker):
        clean = {k: v for k, v in options.items() if k in ("size", "overlap_sentences")}
    elif cls is RecursiveChunker:
        clean = {k: v for k, v in options.items() if k in ("size", "separators")}
    return cast("Chunker", cls(**clean))
