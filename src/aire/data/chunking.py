"""Text chunkers used by RAG ingestion and context packing."""

from __future__ import annotations

import re
from typing import Protocol, cast, runtime_checkable

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


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n{2,}")


class SentenceChunker:
    """Sentence-aware chunker that packs sentences up to ``size`` characters."""

    def __init__(self, size: int = 800, overlap_sentences: int = 1) -> None:
        self.size = size
        self.overlap_sentences = overlap_sentences

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


_CHUNKERS: dict[str, type] = {
    "fixed": FixedChunker,
    "sentence": SentenceChunker,
    "semantic": SentenceChunker,  # semantic boundary approximation via sentences
    "recursive": RecursiveChunker,
}


def get_chunker(name: str = "recursive", **options: object) -> Chunker:
    """Resolve a chunker by name."""
    try:
        cls = _CHUNKERS[name]
    except KeyError:
        from aire.core.errors import NotFoundError

        raise NotFoundError("chunker", name, context={"available": sorted(_CHUNKERS)}) from None
    return cast("Chunker", cls(**options))
