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
    """LLM prompt scorer for (query, passage) pairs — not a HuggingFace CrossEncoder.

    Prefer :class:`HFCrossEncoderReranker` (``reranker="hf_cross_encoder"``) for a
    real cross-encoder. Offline-friendly: when the model echoes the prompt, falls
    back to lexical overlap so tests stay deterministic without a judge model.
    """

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        if not hits:
            return []
        import asyncio

        async def _score(hit: ScoredChunk) -> ScoredChunk:
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
                q_terms = set(tokenize(query))
                c_terms = set(tokenize(hit.chunk.text))
                relevance = (len(q_terms & c_terms) / len(q_terms)) if q_terms else 0.0
            return ScoredChunk(chunk=hit.chunk, score=hit.score + self.weight * relevance)

        rescored = list(await asyncio.gather(*[_score(h) for h in hits]))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]


class HFCrossEncoderReranker:
    """Real HuggingFace CrossEncoder reranker via ``sentence-transformers``.

    Requires ``pip install 'aire[eval]'``. Default model:
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        weight: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.weight = weight
        self._encoder: Any = None

    def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            from aire.core.errors import ConfigurationError

            raise ConfigurationError(
                "HFCrossEncoderReranker requires sentence-transformers: "
                "pip install 'aire[eval]'",
                code="rag.cross_encoder_missing",
            ) from exc
        self._encoder = CrossEncoder(self.model_name)
        return self._encoder

    async def rerank(self, query: str, hits: list[ScoredChunk], *, k: int) -> list[ScoredChunk]:
        if not hits:
            return []
        import asyncio

        encoder = self._load()
        pairs = [(query, hit.chunk.text) for hit in hits]

        def _predict() -> list[float]:
            scores = encoder.predict(pairs)
            return [float(s) for s in scores]

        scores = await asyncio.to_thread(_predict)
        rescored = [
            ScoredChunk(chunk=hit.chunk, score=hit.score + self.weight * score)
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k]


_RERANKERS: dict[str, type] = {
    "none": IdentityReranker,
    "identity": IdentityReranker,
    "lexical": LexicalOverlapReranker,
    "embedding": EmbeddingReranker,
    "model": ModelReranker,
    "hf_cross_encoder": HFCrossEncoderReranker,
    # Honest alias: real CE when no LLM model is passed.
    "cross_encoder": HFCrossEncoderReranker,
}


def get_reranker(name: str = "none", **options: Any) -> Reranker:
    try:
        cls = _RERANKERS[name]
    except KeyError:
        from aire.core.errors import NotFoundError

        raise NotFoundError("reranker", name, context={"available": sorted(_RERANKERS)}) from None
    # Allow callers who still pass model= for the old LLM scorer.
    if name == "cross_encoder" and "model" in options and "model_name" not in options:
        model = options.pop("model")
        if not isinstance(model, str):
            return ModelReranker(model, **options)
    instance = cls(**options)
    assert isinstance(instance, Reranker)
    return instance


def register_reranker(name: str, cls: type, *, replace: bool = False) -> None:
    if name in _RERANKERS and not replace:
        from aire.core.errors import PluginError

        raise PluginError(f"reranker {name!r} already registered", code="registry.duplicate")
    _RERANKERS[name] = cls
