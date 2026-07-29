"""Context compression — pack retrieved chunks into a token budget."""

from __future__ import annotations

from typing import Any, Protocol

from aire.models.base import Model
from aire.models.types import GenerationRequest
from aire.rag.types import ScoredChunk


class ContextCompressor(Protocol):
    async def compress(
        self, query: str, hits: list[ScoredChunk], *, max_chars: int = 4000
    ) -> str: ...

    def describe(self) -> dict[str, Any]: ...


class TruncateCompressor:
    """Greedy pack by score until ``max_chars`` (offline, deterministic)."""

    async def compress(
        self, query: str, hits: list[ScoredChunk], *, max_chars: int = 4000
    ) -> str:
        _ = query
        parts: list[str] = []
        used = 0
        for i, hit in enumerate(hits):
            block = f"[{i + 1}] {hit.chunk.text}"
            if used + len(block) + 2 > max_chars and parts:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    def describe(self) -> dict[str, Any]:
        return {"kind": "context_compressor", "name": "truncate"}


class ExtractiveCompressor:
    """Lexical extractive compression: keep sentences that share tokens with the query.

    Offline and deterministic — not model-based extractive summarization.
    """

    async def compress(
        self, query: str, hits: list[ScoredChunk], *, max_chars: int = 4000
    ) -> str:
        from aire.rag.store import tokenize

        q_tokens = set(tokenize(query))
        parts: list[str] = []
        used = 0
        for i, hit in enumerate(hits):
            sentences = [
                s.strip() for s in hit.chunk.text.replace("?", ".").split(".") if s.strip()
            ]
            kept = [
                s for s in sentences if q_tokens & set(tokenize(s)) or not q_tokens
            ] or sentences[:1]
            block = f"[{i + 1}] " + ". ".join(kept)
            if used + len(block) + 2 > max_chars and parts:
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    def describe(self) -> dict[str, Any]:
        return {"kind": "context_compressor", "name": "lexical_extractive"}


class ModelCompressor:
    """Ask a model to compress retrieved context for the question."""

    def __init__(self, model: Model) -> None:
        self.model = model

    async def compress(
        self, query: str, hits: list[ScoredChunk], *, max_chars: int = 4000
    ) -> str:
        raw = "\n\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(hits))
        if len(raw) <= max_chars:
            return raw
        prompt = (
            f"Compress the context to under {max_chars} characters while keeping "
            f"facts needed to answer: {query}\n\nContext:\n{raw[: max_chars * 3]}\n\n"
            "Compressed:"
        )
        text = (await self.model.generate(GenerationRequest.of(prompt))).text.strip()
        return text[:max_chars]

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "context_compressor",
            "name": "model",
            "model": self.model.info.ref,
        }


def get_compressor(name: str, *, model: Model | None = None) -> ContextCompressor:
    from aire.core.errors import ConfigurationError

    if name == "truncate":
        return TruncateCompressor()
    if name == "extractive":
        return ExtractiveCompressor()
    if name == "model":
        if model is None:
            raise ConfigurationError(
                "compressor 'model' requires a model= argument",
                code="rag.compressor_model_required",
            )
        return ModelCompressor(model)
    raise ConfigurationError(
        f"unknown compressor {name!r}",
        code="rag.compressor_unknown",
        context={"available": ["truncate", "extractive", "model"]},
    )
