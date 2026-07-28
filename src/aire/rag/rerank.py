"""Rerankers: second-pass ordering of retrieved candidates."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aire.rag.store import cosine_similarity, tokenize
from aire.rag.types import ScoredChunk

if TYPE_CHECKING:
    from aire.models.base import EmbeddingModel, Model


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


class EmbeddingReranker:
    """Re-score hits by cosine similarity between query and chunk embeddings."""

    def __init__(self, embedder: EmbeddingModel, *, weight: float = 1.0) -> None:
        self.embedder = embedder
        self.weight = weight

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        if not hits:
            return []
        query_vec = await self.embedder.embed_one(query)
        rescored: list[ScoredChunk] = []
        for hit in hits:
            chunk_vec = hit.chunk.embedding
            if not chunk_vec:
                chunk_vec = await self.embedder.embed_one(hit.chunk.text)
            sim = cosine_similarity(query_vec, chunk_vec)
            rescored.append(ScoredChunk(chunk=hit.chunk, score=hit.score + self.weight * sim))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]


_SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


class ModelReranker:
    """Cross-encoder-style rerank: ask a model to score each (query, passage) pair.

    Offline-friendly: when the model echoes the prompt, falls back to lexical
    overlap so tests and CI stay deterministic without a real judge model.
    """

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        if not hits:
            return []
        rescored: list[ScoredChunk] = []
        for hit in hits:
            prompt = (
                "Score how relevant the PASSAGE is to the QUERY on a 0-10 scale. "
                "Respond with only the number.\n"
                f"QUERY: {query}\nPASSAGE: {hit.chunk.text[:1200]}\nSCORE:"
            )
            text = await self.model.ask(prompt, max_tokens=8)
            match = _SCORE_RE.search(text or "")
            if match:
                raw = float(match.group(1))
                relevance = raw if raw <= 1.0 else min(10.0, max(0.0, raw)) / 10.0
            else:
                # deterministic offline fallback
                q_terms = set(tokenize(query))
                c_terms = set(tokenize(hit.chunk.text))
                relevance = (len(q_terms & c_terms) / len(q_terms)) if q_terms else 0.0
            rescored.append(
                ScoredChunk(chunk=hit.chunk, score=hit.score + self.weight * relevance)
            )
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]


_RERANKERS: dict[str, type] = {
    "none": IdentityReranker,
    "identity": IdentityReranker,
    "lexical": LexicalOverlapReranker,
    "embedding": EmbeddingReranker,
    "model": ModelReranker,
    "cross_encoder": ModelReranker,
}


def get_reranker(name: str = "none", **options: Any) -> Reranker:
    try:
        cls = _RERANKERS[name]
    except KeyError:
        from aire.core.errors import NotFoundError

        raise NotFoundError("reranker", name, context={"available": sorted(_RERANKERS)}) from None
    instance = cls(**options)
    assert isinstance(instance, Reranker)
    return instance


def register_reranker(name: str, cls: type, *, replace: bool = False) -> None:
    if name in _RERANKERS and not replace:
        from aire.core.errors import PluginError

        raise PluginError(f"reranker {name!r} already registered", code="registry.duplicate")
    _RERANKERS[name] = cls
