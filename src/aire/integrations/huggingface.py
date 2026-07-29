"""Hugging Face provider (``"huggingface:<model>"``).

Uses the Hugging Face Inference API (serverless, OpenAI-compatible router) so
no heavy local dependencies are required. Local ``transformers`` pipelines can
be exposed through ``callable:<name>`` instead, keeping torch out of core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.core.errors import AuthenticationError
from aire.core.plugins import PluginInfo
from aire.integrations.http import ProviderHttpClient
from aire.integrations.openai import OpenAIEmbedder, OpenAIModel
from aire.models.base import EmbeddingModel, Model

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

ROUTER_BASE_URL = "https://router.huggingface.co/v1"


def register(runtime: Runtime) -> PluginInfo:
    def _client(runtime: Runtime, options: dict[str, Any]) -> ProviderHttpClient:
        cred = runtime.settings.credential("huggingface")
        api_key = options.get("api_key") or cred.resolve_key("HF_TOKEN")
        if not api_key:
            raise AuthenticationError(
                "huggingface",
                "no token: set HF_TOKEN, providers.huggingface.api_key, or pass api_key=",
            )
        base_url = options.get("base_url") or cred.base_url or ROUTER_BASE_URL
        return ProviderHttpClient(
            runtime,
            "huggingface",
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", **cred.default_headers},
        )

    def _model_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        return OpenAIModel(name, _client(runtime, options), provider="huggingface")

    def _embedder_factory(name: str, *, runtime: Runtime, **options: Any) -> EmbeddingModel:
        return OpenAIEmbedder(name, _client(runtime, options), provider="huggingface")

    runtime.model_providers.register("huggingface", _model_factory, replace=True)
    runtime.embedders.register("huggingface", _embedder_factory, replace=True)
    return PluginInfo(
        name="huggingface",
        version="0.1.0",
        provides=["model:huggingface", "embedder:huggingface"],
    )


class HuggingFaceProvider:
    """Entry-point target for the ``huggingface`` provider."""

    @staticmethod
    def register(runtime: Runtime) -> PluginInfo:
        return register(runtime)
