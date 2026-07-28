"""Universal OpenAI-compatible provider with named endpoint aliases.

Most inference servers — local ones (LM Studio, llama.cpp, vLLM, MLX,
llamafile, LocalAI, TGI) and hosted APIs (Groq, Together, Fireworks, DeepSeek,
Mistral, xAI, OpenRouter, Cerebras, Perplexity, Azure proxies) — speak the
OpenAI chat-completions protocol. This module gives each of them a first-class
``provider:name`` reference with a sensible default base URL, so no per-vendor
adapter is ever needed::

    AI.models.use_sync("lmstudio:qwen2.5-7b-instruct")      # local, no key
    AI.models.use_sync("llamacpp:llama-3.2")                # llama.cpp server
    AI.models.use_sync("vllm:meta-llama/Llama-3.1-8B")      # vLLM server
    AI.models.use_sync("groq:llama-3.3-70b-versatile")      # hosted, GROQ_API_KEY
    AI.models.use_sync(
        "openai_compatible:my-model",
        base_url="http://gpu-box:8000/v1",
        api_key="...",
    )                                                        # anything else

Configuration (by priority): explicit options → ``aire.yaml providers.<name>``
→ provider-specific env vars (``GROQ_API_KEY``, ``LMSTUDIO_BASE_URL``, ...) →
the alias defaults below.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from aire.core.errors import AuthenticationError, ConfigurationError
from aire.integrations.http import ProviderHttpClient
from aire.integrations.openai import OpenAIEmbedder, OpenAIModel
from aire.models.base import EmbeddingModel, Model

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


class EndpointSpec(BaseModel):
    """Describes one known OpenAI-compatible endpoint — agent-inspectable."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    env_key: str | None = None
    env_base_url: str | None = None
    requires_api_key: bool = False
    local: bool = False
    description: str = ""


KNOWN_ENDPOINTS: dict[str, EndpointSpec] = {
    # -- local inference servers (no key required by default) -------------------
    "lmstudio": EndpointSpec(
        base_url="http://localhost:1234/v1",
        env_key="LMSTUDIO_API_KEY",
        env_base_url="LMSTUDIO_BASE_URL",
        local=True,
        description="LM Studio desktop server (GGUF and MLX models)",
    ),
    "llamacpp": EndpointSpec(
        base_url="http://localhost:8080/v1",
        env_key="LLAMACPP_API_KEY",
        env_base_url="LLAMACPP_BASE_URL",
        local=True,
        description="llama.cpp server (GGUF models, CPU/GPU)",
    ),
    "llamafile": EndpointSpec(
        base_url="http://localhost:8080/v1",
        env_base_url="LLAMAFILE_BASE_URL",
        local=True,
        description="llamafile single-file model server",
    ),
    "vllm": EndpointSpec(
        base_url="http://localhost:8000/v1",
        env_key="VLLM_API_KEY",
        env_base_url="VLLM_BASE_URL",
        local=True,
        description="vLLM high-throughput serving",
    ),
    "mlx": EndpointSpec(
        base_url="http://localhost:8080/v1",
        env_key="MLX_API_KEY",
        env_base_url="MLX_BASE_URL",
        local=True,
        description="mlx-lm server (Apple Silicon models)",
    ),
    "localai": EndpointSpec(
        base_url="http://localhost:8080/v1",
        env_key="LOCALAI_API_KEY",
        env_base_url="LOCALAI_BASE_URL",
        local=True,
        description="LocalAI drop-in OpenAI replacement",
    ),
    "tgi": EndpointSpec(
        base_url="http://localhost:8080/v1",
        env_key="TGI_API_KEY",
        env_base_url="TGI_BASE_URL",
        local=True,
        description="Hugging Face text-generation-inference server",
    ),
    # -- hosted OpenAI-compatible APIs ------------------------------------------
    "groq": EndpointSpec(
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        env_base_url="GROQ_BASE_URL",
        requires_api_key=True,
        description="Groq LPU inference (Llama, Mixtral, ...)",
    ),
    "together": EndpointSpec(
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        env_base_url="TOGETHER_BASE_URL",
        requires_api_key=True,
        description="Together AI hosted open models",
    ),
    "fireworks": EndpointSpec(
        base_url="https://api.fireworks.ai/inference/v1",
        env_key="FIREWORKS_API_KEY",
        env_base_url="FIREWORKS_BASE_URL",
        requires_api_key=True,
        description="Fireworks AI hosted open models",
    ),
    "deepseek": EndpointSpec(
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        env_base_url="DEEPSEEK_BASE_URL",
        requires_api_key=True,
        description="DeepSeek API",
    ),
    "mistral": EndpointSpec(
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        env_base_url="MISTRAL_BASE_URL",
        requires_api_key=True,
        description="Mistral AI API (La Plateforme)",
    ),
    "xai": EndpointSpec(
        base_url="https://api.x.ai/v1",
        env_key="XAI_API_KEY",
        env_base_url="XAI_BASE_URL",
        requires_api_key=True,
        description="xAI Grok API",
    ),
    "openrouter": EndpointSpec(
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        env_base_url="OPENROUTER_BASE_URL",
        requires_api_key=True,
        description="OpenRouter multi-provider hub",
    ),
    "cerebras": EndpointSpec(
        base_url="https://api.cerebras.ai/v1",
        env_key="CEREBRAS_API_KEY",
        env_base_url="CEREBRAS_BASE_URL",
        requires_api_key=True,
        description="Cerebras wafer-scale inference",
    ),
    "perplexity": EndpointSpec(
        base_url="https://api.perplexity.ai",
        env_key="PERPLEXITY_API_KEY",
        env_base_url="PERPLEXITY_BASE_URL",
        requires_api_key=True,
        description="Perplexity online LLM API",
    ),
}

#: Provider name for arbitrary, unlisted OpenAI-compatible servers.
GENERIC_PROVIDER = "openai_compatible"


def describe_endpoints() -> dict[str, Any]:
    """Machine-readable catalog of known endpoints — for agents and docs."""
    return {name: spec.model_dump(mode="json") for name, spec in sorted(KNOWN_ENDPOINTS.items())}


def _resolve_client(
    runtime: Runtime, provider: str, spec: EndpointSpec | None, options: dict[str, Any]
) -> ProviderHttpClient:
    cred = runtime.settings.credential(provider)
    base_url = (
        options.get("base_url")
        or cred.base_url
        or (os.environ.get(spec.env_base_url) if spec and spec.env_base_url else None)
        or (spec.base_url if spec else None)
    )
    if not base_url:
        raise ConfigurationError(
            f"{GENERIC_PROVIDER} requires a base_url: pass base_url=... or set "
            f"providers.{GENERIC_PROVIDER}.base_url in aire.yaml",
            code="config.missing_base_url",
            context={"provider": provider},
        )
    api_key = (
        options.get("api_key")
        or (cred.api_key.get_secret_value() if cred.api_key else None)
        or (os.environ.get(spec.env_key) if spec and spec.env_key else None)
    )
    if spec is not None and spec.requires_api_key and not api_key:
        raise AuthenticationError(
            provider,
            f"no API key: set {spec.env_key}, providers.{provider}.api_key, or pass api_key=",
        )
    headers = dict(cred.default_headers)
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    return ProviderHttpClient(runtime, provider, base_url=base_url, headers=headers)


def register(runtime: Runtime) -> None:
    """Register every known alias (plus the generic provider) on a runtime."""

    def _model_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        provider = options.pop("provider", GENERIC_PROVIDER)
        spec = KNOWN_ENDPOINTS.get(provider)
        client = _resolve_client(runtime, provider, spec, options)
        return OpenAIModel(name, client, provider=provider)

    def _embedder_factory(name: str, *, runtime: Runtime, **options: Any) -> EmbeddingModel:
        provider = options.pop("provider", GENERIC_PROVIDER)
        spec = KNOWN_ENDPOINTS.get(provider)
        client = _resolve_client(runtime, provider, spec, options)
        return OpenAIEmbedder(name, client, provider=provider)

    for alias in KNOWN_ENDPOINTS:
        runtime.model_providers.register(
            alias,
            lambda name, *, runtime, _a=alias, **options: _model_factory(
                name, runtime=runtime, provider=_a, **options
            ),
            replace=True,
        )
        runtime.embedders.register(
            alias,
            lambda name, *, runtime, _a=alias, **options: _embedder_factory(
                name, runtime=runtime, provider=_a, **options
            ),
            replace=True,
        )
    runtime.model_providers.register(GENERIC_PROVIDER, _model_factory, replace=True)
    runtime.embedders.register(GENERIC_PROVIDER, _embedder_factory, replace=True)
