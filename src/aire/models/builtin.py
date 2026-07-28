"""Zero-dependency builtin providers.

These make the library fully usable offline — for tests, development, CI and as
reference implementations of the provider contract:

- ``mock:echo`` — deterministic echo model with tool-call scripting support.
- ``local:hashing`` — deterministic feature-hashing embedder (no downloads).
- ``callable:<registered_name>`` — wrap any Python callable as a model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aire.core.types import Capability, HealthStatus, Usage
from aire.models.base import EmbeddingModel, Model, estimate_tokens
from aire.models.types import (
    EmbeddingRequest,
    EmbeddingResult,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ToolCall,
)

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


class EchoModel(Model):
    """Deterministic model that echoes its input.

    Useful beyond testing: it lets the entire stack (agents, RAG, evaluation,
    tracing, deployment) run end-to-end with zero credentials and zero cost.

    If the last user message contains embedded context (RAG style), the echo
    includes the leading context lines, which keeps citation flows testable.
    """

    def __init__(self, name: str = "echo", *, prefix: str = "") -> None:
        self._name = name
        self._prefix = prefix
        self.scripted_tool_calls: list[ToolCall] | None = None

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            ref=f"mock:{self._name}",
            provider="mock",
            capabilities=[
                Capability.TEXT_GENERATION,
                Capability.STREAMING,
                Capability.STRUCTURED_OUTPUT,
                Capability.TOOL_CALLING,
            ],
            context_window=128_000,
            latency_ms_p50=0.1,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt_tokens = sum(estimate_tokens(m.text_content) for m in request.messages)
        if request.response_format is not None:
            text = _structured_stub(request.response_format.json_schema)
        elif self.scripted_tool_calls:
            result = GenerationResult(
                content=[],
                tool_calls=self.scripted_tool_calls,
                finish_reason="tool_calls",
                model=self.info.ref,
                usage=Usage(input_tokens=prompt_tokens, output_tokens=8),
            )
            self.scripted_tool_calls = None
            return result
        else:
            last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
            body = last_user.text_content if last_user else ""
            text = f"{self._prefix}{body}"
        return GenerationResult.text_result(
            text,
            model=self.info.ref,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=estimate_tokens(text)),
        )

    async def health(self) -> HealthStatus:
        return HealthStatus.healthy("mock provider always available", latency_ms=0.0)


def _structured_stub(schema: dict[str, Any], *, _depth: int = 0) -> str:
    """Emit minimal valid JSON for a schema (for offline structured tests)."""
    import json

    defs = dict(schema.get("$defs", {}))
    defs.update(schema.get("definitions", {}))
    return json.dumps(_stub_value(schema, 0, defs))


_STUB_SCALARS: dict[str, Any] = {
    "string": "mock-value",
    "integer": 0,
    "number": 0.0,
    "boolean": False,
}


def _stub_value(schema: dict[str, Any], depth: int, defs: dict[str, Any]) -> Any:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        schema = defs.get(ref.rsplit("/", 1)[-1], schema)
    ptype = schema.get("type", "object" if "properties" in schema else "string")
    if ptype in _STUB_SCALARS:
        return _STUB_SCALARS[ptype]
    if depth >= 3:
        return [] if ptype == "array" else {} if ptype == "object" else None
    if ptype == "array":
        items = schema.get("items", {"type": "string"})
        return [_stub_value(items, depth + 1, defs), _stub_value(items, depth + 1, defs)]
    if ptype == "object":
        return {
            key: _stub_value(prop, depth + 1, defs)
            for key, prop in schema.get("properties", {}).items()
        }
    return None


class CallableModel(Model):
    """Wrap a Python callable ``(prompt: str) -> str | Awaitable[str]`` as a Model."""

    def __init__(
        self,
        name: str,
        fn: Callable[[str], str | Awaitable[str]],
        *,
        context_window: int | None = None,
    ) -> None:
        self._name = name
        self._fn = fn
        self._context_window = context_window

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            ref=f"callable:{self._name}",
            provider="callable",
            capabilities=[Capability.TEXT_GENERATION],
            context_window=self._context_window,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        import inspect

        prompt = "\n".join(m.text_content for m in request.messages if m.role != "system")
        system = "\n".join(m.text_content for m in request.messages if m.role == "system")
        full_prompt = f"{system}\n{prompt}".strip()
        outcome = self._fn(full_prompt)
        text = await outcome if inspect.isawaitable(outcome) else outcome
        return GenerationResult.text_result(
            str(text),
            model=self.info.ref,
            usage=Usage(
                input_tokens=estimate_tokens(full_prompt),
                output_tokens=estimate_tokens(str(text)),
            ),
        )


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(EmbeddingModel):
    """Deterministic feature-hashing embedder with no external dependencies.

    Tokens are hashed into ``dimension`` buckets with signed counts, then L2
    normalized. Not semantically strong, but stable, fast, offline, and good
    enough for lexical retrieval, caching and tests. Swap in a provider
    embedder (``openai:text-embedding-3-small``) for production semantics.
    """

    def __init__(self, name: str = "hashing", *, dimension: int = 256) -> None:
        self._name = name
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"local:{self._name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = [self._embed_text(t) for t in request.inputs]
        return EmbeddingResult(vectors=vectors, model=self.name)

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def health(self) -> HealthStatus:
        return HealthStatus.healthy("hashing embedder is offline and always available")


def register_builtins(runtime: Runtime) -> None:
    """Register builtin factories on a runtime. Called by ``Runtime.__init__``."""

    def _mock_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        return EchoModel(name, **options)

    def _callable_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        from aire.models.registry import _CALLABLES

        if name not in _CALLABLES:
            from aire.core.errors import NotFoundError

            raise NotFoundError(
                "callable model",
                name,
                context={"hint": "register one with AI.models.register_callable(name, fn)"},
            )
        return CallableModel(name, _CALLABLES[name], **options)

    def _hashing_factory(name: str, *, runtime: Runtime, **options: Any) -> EmbeddingModel:
        return HashingEmbedder(name, **options)

    runtime.model_providers.register("mock", _mock_factory, replace=True)
    runtime.model_providers.register("callable", _callable_factory, replace=True)
    runtime.embedders.register("local", _hashing_factory, replace=True)
