"""Rerankers: second-pass ordering of retrieved candidates."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aire.rag.store import tokenize
from aire.rag.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]: ...


class IdentityReranker:
    """Pass-through reranker (default)."""

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        return hits[:k]


class LexicalOverlapReranker:
    """Boosts hits by query-term coverage — cheap, deterministic, offline."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not terms:
            return hits[:k]
        rescored: list[ScoredChunk] = []
        for hit in hits:
            chunk_terms = set(tokenize(hit.chunk.text))
            coverage = len(terms & chunk_terms) / len(terms)
            rescored.append(ScoredChunk(chunk=hit.chunk, score=hit.score + self.weight * coverage))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]


_RERANKERS: dict[str, type] = {
    "none": IdentityReranker,
    "identity": IdentityReranker,
    "lexical": LexicalOverlapReranker,
}


def get_reranker(name: str = "none", **options: object) -> Reranker:
    try:
        cls = _RERANKERS[name]
    except KeyError:
        from aire.core.errors import NotFoundError

        raise NotFoundError("reranker", name, context={"available": sorted(_RERANKERS)}) from None
    instance = cls(**options)
    assert isinstance(instance, Reranker)
    return instance
