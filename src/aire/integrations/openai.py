"""OpenAI-compatible chat + embeddings provider.

Works against OpenAI itself and any API-compatible endpoint (vLLM, llama.cpp,
LM Studio, Together, Groq, Azure with a compatible proxy, ...): point
``base_url`` at the server. Activate with ``register(runtime)`` or implicitly
by using ``"openai:<model>"``.

Configuration (by priority): explicit options → ``aire.yaml providers.openai``
→ ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from aire.core.content import Message
from aire.core.errors import AuthenticationError
from aire.core.plugins import PluginInfo
from aire.core.types import HealthStatus, Usage
from aire.integrations.http import ProviderHttpClient
from aire.models.base import EmbeddingModel, Model
from aire.models.retry import with_retry
from aire.models.types import (
    EmbeddingRequest,
    EmbeddingResult,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ToolCall,
)

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Pricing USD per million tokens for well-known models (best-effort defaults).
_KNOWN: dict[str, dict[str, Any]] = {
    "gpt-4o": {"in": 2.50, "out": 10.00, "ctx": 128_000},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60, "ctx": 128_000},
    "gpt-4.1": {"in": 2.00, "out": 8.00, "ctx": 1_000_000},
    "text-embedding-3-small": {"in": 0.02, "ctx": 8_191},
    "text-embedding-3-large": {"in": 0.13, "ctx": 8_191},
}


class OpenAIModel(Model):
    """Model backed by an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        name: str,
        client: ProviderHttpClient,
        *,
        provider: str = "openai",
        cost: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._client = client
        self._provider = provider
        known = _KNOWN.get(name, {})
        self._cost = cost or known

    @property
    def info(self) -> ModelInfo:
        from aire.integrations.openai_media import capabilities_for_openai_model
        from aire.models.types import CostInfo

        return ModelInfo(
            ref=f"{self._provider}:{self._name}",
            provider=self._provider,
            capabilities=capabilities_for_openai_model(self._name),
            context_window=self._cost.get("ctx"),
            cost=CostInfo(
                input_per_million=self._cost.get("in"),
                output_per_million=self._cost.get("out"),
            ),
        )

    # -- payload conversion ---------------------------------------------------------

    def _messages_payload(self, messages: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.text_content}
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            payload.append(entry)
        return payload

    def _request_payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._name,
            "messages": self._messages_payload(request.messages),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_format.name,
                    "schema": request.response_format.json_schema,
                    "strict": request.response_format.strict,
                },
            }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- interface --------------------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._request_payload(request, stream=False)

        async def _call() -> dict[str, Any]:
            return await self._client.post_json("/chat/completions", payload)

        data = await with_retry(_call)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_raw = data.get("usage", {}) or {}
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
        )
        usage = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self.info.cost.estimate(usage),
        )
        tool_calls = [
            ToolCall.from_json(
                tc.get("id", ""),
                tc.get("function", {}).get("name", ""),
                tc.get("function", {}).get("arguments", "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        finish = choice.get("finish_reason") or "stop"
        return GenerationResult(
            content=[_text(message.get("content") or "")],
            tool_calls=tool_calls,
            finish_reason=_finish(finish),
            model=self.info.ref,
            usage=usage,
            raw={"provider_response_id": data.get("id")},
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        payload = self._request_payload(request, stream=True)
        async for data in self._client.stream_sse("/chat/completions", payload):
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content") or ""
            finish = choices[0].get("finish_reason")
            calls = [
                ToolCall.from_json(
                    tc.get("id", ""),
                    tc.get("function", {}).get("name", ""),
                    tc.get("function", {}).get("arguments", "{}"),
                )
                for tc in delta.get("tool_calls") or []
                if tc.get("function", {}).get("name")
            ]
            yield GenerationChunk(
                text=text, tool_calls=calls, finish_reason=_finish(finish) if finish else None
            )

    async def health(self) -> HealthStatus:
        try:
            await self._client.get_json("/models")
        except AuthenticationError as exc:
            return HealthStatus.unhealthy(str(exc))
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()


class OpenAIEmbedder(EmbeddingModel):
    """OpenAI-compatible embeddings endpoint."""

    def __init__(self, name: str, client: ProviderHttpClient, *, provider: str = "openai") -> None:
        self._name = name
        self._client = client
        self._provider = provider
        self._dimension = 1536 if "small" in name or "ada" in name else 3072

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        async def _call() -> dict[str, Any]:
            return await self._client.post_json(
                "/embeddings", {"model": self._name, "input": request.inputs}
            )

        data = await with_retry(_call)
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        usage_raw = data.get("usage", {}) or {}
        return EmbeddingResult(
            vectors=[row.get("embedding", []) for row in rows],
            model=self.name,
            usage=Usage(input_tokens=usage_raw.get("prompt_tokens", 0)),
        )


def _text(content: str) -> Any:
    from aire.core.content import TextContent

    return TextContent(text=content)


def _finish(reason: str | None) -> Any:
    mapping = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(reason or "stop", "stop")


def _resolve_client(runtime: Runtime, provider: str, options: dict[str, Any]) -> ProviderHttpClient:
    cred = runtime.settings.credential(provider)
    api_key = options.get("api_key") or cred.resolve_key("OPENAI_API_KEY")
    base_url = (
        options.get("base_url") or cred.base_url or _env("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    )
    headers = dict(cred.default_headers)
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    elif provider == "openai" and "api.openai.com" in base_url:
        raise AuthenticationError(
            provider,
            "no API key: set OPENAI_API_KEY, providers.openai.api_key, or pass api_key=",
        )
    return ProviderHttpClient(runtime, provider, base_url=base_url, headers=headers)


def _env(name: str) -> str | None:
    import os

    return os.environ.get(name)


def register(runtime: Runtime) -> PluginInfo:
    """Register the OpenAI-compatible provider on a runtime."""

    def _model_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        provider = options.pop("provider", "openai")
        client = _resolve_client(runtime, provider, options)
        return OpenAIModel(name, client, provider=provider)

    def _embedder_factory(name: str, *, runtime: Runtime, **options: Any) -> EmbeddingModel:
        client = _resolve_client(runtime, "openai", options)
        return OpenAIEmbedder(name, client)

    runtime.model_providers.register("openai", _model_factory, replace=True)
    runtime.embedders.register("openai", _embedder_factory, replace=True)
    return PluginInfo(
        name="openai",
        version="0.1.0",
        provides=["model:openai", "embedder:openai"],
    )


class OpenAIProvider:
    """Entry-point target for the ``openai`` provider."""

    @staticmethod
    def register(runtime: Runtime) -> PluginInfo:
        return register(runtime)
