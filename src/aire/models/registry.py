"""Model resolution: turn ``"provider:name"`` strings into live models."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from aire.core.errors import ProviderError
from aire.core.types import Ref
from aire.models.base import EmbeddingModel, Model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aire.core.runtime import Runtime

# Process-wide registry of Python callables exposed as ``callable:<name>`` models.
_CALLABLES: dict[str, Callable[[str], str | Awaitable[str]]] = {}


def register_callable(name: str, fn: Callable[[str], str | Awaitable[str]]) -> None:
    """Expose a Python function as a model addressable via ``callable:<name>``."""
    _CALLABLES[name] = fn


class ModelRegistry:
    """Resolves :class:`Ref` identifiers against registered provider factories.

    Provider factories have the signature::

        factory(name: str, *, runtime: Runtime, **options) -> Model

    Instances are cached so repeated ``use("openai:gpt-4o-mini")`` calls share
    one underlying HTTP client (and its connection pool).
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._models: dict[str, Model] = {}
        self._embedders: dict[str, EmbeddingModel] = {}
        self._lock = asyncio.Lock()

    async def use(self, spec: str | Ref, /, *, cache: bool = True, **options: Any) -> Model:
        """Resolve a model by reference, e.g. ``openai:gpt-4o-mini``."""
        ref = Ref.parse(spec)
        key = str(ref)
        if cache and key in self._models:
            return self._models[key]
        async with self._lock:
            if cache and key in self._models:
                return self._models[key]
            if not self._runtime.model_providers.has(ref.provider):
                _maybe_hint_integration(ref.provider, self._runtime)
            factory = self._runtime.model_providers.get_factory(ref.provider)
            model = factory(ref.name, runtime=self._runtime, **options)
            if not isinstance(model, Model):
                raise ProviderError(
                    ref.provider,
                    f"factory for {key!r} returned {type(model).__name__}, expected Model",
                    retryable=False,
                )
            if cache:
                self._models[key] = model
            return model

    async def embedder(
        self, spec: str | Ref | None = None, /, *, cache: bool = True, **options: Any
    ) -> EmbeddingModel:
        """Resolve an embedding model; defaults to the configured embedder."""
        if spec is None:
            spec = self._runtime.settings.model.embedder
        ref = Ref.parse(spec)
        key = str(ref)
        if cache and key in self._embedders:
            return self._embedders[key]
        async with self._lock:
            if cache and key in self._embedders:
                return self._embedders[key]
            if not self._runtime.embedders.has(ref.provider):
                _maybe_hint_integration(ref.provider, self._runtime)
            factory = self._runtime.embedders.get_factory(ref.provider)
            embedder = factory(ref.name, runtime=self._runtime, **options)
            if not isinstance(embedder, EmbeddingModel):
                raise ProviderError(
                    ref.provider,
                    f"factory for {key!r} returned {type(embedder).__name__}, "
                    "expected EmbeddingModel",
                    retryable=False,
                )
            if cache:
                self._embedders[key] = embedder
            return embedder

    def registered_providers(self) -> list[str]:
        return self._runtime.model_providers.names()

    def describe(self) -> dict[str, Any]:
        return {
            "providers": self.registered_providers(),
            "embedders": self._runtime.embedders.names(),
            "cached_models": sorted(self._models),
        }


# Provider prefix → integration module with a ``register(runtime)`` function.
_INTEGRATION_MODULES: dict[str, str] = {
    "openai": "aire.integrations.openai",
    "anthropic": "aire.integrations.anthropic",
    "ollama": "aire.integrations.ollama",
    "huggingface": "aire.integrations.huggingface",
    "qdrant": "aire.integrations.qdrant",
    "chroma": "aire.integrations.chroma",
    "pinecone": "aire.integrations.pinecone",
    "weaviate": "aire.integrations.weaviate",
    "milvus": "aire.integrations.milvus",
    "sqlite": "aire.rag.sqlite",
}

# OpenAI-compatible endpoints (local servers and hosted APIs) all live in one
# module; touching any single alias registers the whole catalog.
for _alias in (
    "openai_compatible",
    "lmstudio",
    "llamacpp",
    "llamafile",
    "vllm",
    "mlx",
    "localai",
    "tgi",
    "groq",
    "together",
    "fireworks",
    "deepseek",
    "mistral",
    "xai",
    "openrouter",
    "cerebras",
    "perplexity",
):
    _INTEGRATION_MODULES[_alias] = "aire.integrations.openai_compat"


def _maybe_hint_integration(provider: str, runtime: Runtime) -> None:
    """Lazily activate a bundled first-party integration, if one exists."""
    import importlib

    module_name = _INTEGRATION_MODULES.get(provider)
    if module_name is None:
        return
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return
    register = getattr(module, "register", None)
    if callable(register):
        register(runtime)
