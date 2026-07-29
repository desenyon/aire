"""Query rewriting strategies for RAG (offline-capable defaults)."""

from __future__ import annotations

from typing import Any, Protocol

from aire.models.base import Model
from aire.models.types import GenerationRequest


class QueryRewriter(Protocol):
    async def rewrite(self, query: str) -> list[str]:
        """Return one or more search queries derived from ``query``."""
        ...

    def describe(self) -> dict[str, Any]: ...


class IdentityRewriter:
    """Pass-through (useful as baseline / disable rewrite)."""

    async def rewrite(self, query: str) -> list[str]:
        return [query]

    def describe(self) -> dict[str, Any]:
        return {"kind": "query_rewriter", "name": "identity"}


class MultiQueryRewriter:
    """Expand a query into N lexical template variants (no model / no embeddings)."""

    def __init__(self, *, n: int = 3) -> None:
        self.n = max(1, n)

    async def rewrite(self, query: str) -> list[str]:
        q = query.strip()
        variants = [q]
        if self.n >= 2:
            variants.append(f"explain {q}")
        if self.n >= 3:
            variants.append(f"key facts about {q}")
        if self.n >= 4:
            variants.append(f"summary of {q}")
        return variants[: self.n]

    def describe(self) -> dict[str, Any]:
        return {"kind": "query_rewriter", "name": "lexical_multi_query", "n": self.n}


class HyDERewriter:
    """Hypothetical Document Embeddings: draft an answer, then search with it."""

    def __init__(self, model: Model, *, keep_original: bool = True) -> None:
        self.model = model
        self.keep_original = keep_original

    async def rewrite(self, query: str) -> list[str]:
        prompt = (
            "Write a short hypothetical passage that would answer this question. "
            "Do not say you lack information — invent a plausible answer.\n\n"
            f"Question: {query}\nPassage:"
        )
        hypo = (await self.model.generate(GenerationRequest.of(prompt))).text.strip()
        out = [query, hypo] if self.keep_original else [hypo]
        return [q for q in out if q]

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "query_rewriter",
            "name": "hyde",
            "model": self.model.info.ref,
            "keep_original": self.keep_original,
        }


class ModelRewriter:
    """Ask a model to emit rewritten search queries (one per line)."""

    def __init__(self, model: Model, *, n: int = 3) -> None:
        self.model = model
        self.n = n

    async def rewrite(self, query: str) -> list[str]:
        prompt = (
            f"Rewrite the search query into {self.n} diverse alternatives. "
            "One query per line, no numbering.\n\n"
            f"Query: {query}"
        )
        text = (await self.model.generate(GenerationRequest.of(prompt))).text
        lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
        if not lines:
            return [query]
        return [query, *lines][: self.n + 1]

    def describe(self) -> dict[str, Any]:
        return {"kind": "query_rewriter", "name": "model", "n": self.n}


def get_rewriter(name: str, *, model: Model | None = None, **options: Any) -> QueryRewriter:
    from aire.core.errors import ConfigurationError

    if name in ("hyde", "model") and model is None:
        raise ConfigurationError(
            f"rewriter {name!r} requires a model= argument",
            code="rag.rewriter_model_required",
            context={"rewriter": name},
        )
    table: dict[str, Any] = {
        "identity": IdentityRewriter,
        "multi_query": MultiQueryRewriter,
        "hyde": lambda **kw: HyDERewriter(model=model, **kw),  # type: ignore[arg-type]
        "model": lambda **kw: ModelRewriter(model=model, **kw),  # type: ignore[arg-type]
    }
    if name not in table:
        raise ConfigurationError(
            f"unknown rewriter {name!r}",
            code="rag.rewriter_unknown",
            context={"available": sorted(table)},
        )
    factory = table[name]
    return factory(**options)  # type: ignore[no-any-return]
