"""Anthropic Messages API provider (``"anthropic:<model>"``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from aire.core.content import TextContent
from aire.core.errors import AuthenticationError
from aire.core.types import Capability, HealthStatus, Usage
from aire.integrations.http import ProviderHttpClient
from aire.models.base import Model
from aire.models.retry import with_retry
from aire.models.types import (
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ToolCall,
)

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"

_KNOWN: dict[str, dict[str, Any]] = {
    "claude-opus-4-1": {"in": 15.0, "out": 75.0, "ctx": 200_000},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0, "ctx": 200_000},
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0, "ctx": 200_000},
}


class AnthropicModel(Model):
    def __init__(self, name: str, client: ProviderHttpClient) -> None:
        self._name = name
        self._client = client
        self._cost = _KNOWN.get(name, {})

    @property
    def info(self) -> ModelInfo:
        from aire.models.types import CostInfo

        return ModelInfo(
            ref=f"anthropic:{self._name}",
            provider="anthropic",
            capabilities=[
                Capability.TEXT_GENERATION,
                Capability.STREAMING,
                Capability.TOOL_CALLING,
                Capability.VISION_INPUT,
            ],
            context_window=self._cost.get("ctx"),
            cost=CostInfo(
                input_per_million=self._cost.get("in"),
                output_per_million=self._cost.get("out"),
            ),
        )

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        system = "\n".join(m.text_content for m in request.messages if m.role == "system")
        messages = [
            {
                "role": m.role if m.role in {"user", "assistant"} else "user",
                "content": m.text_content,
            }
            for m in request.messages
            if m.role != "system"
        ]
        payload: dict[str, Any] = {
            "model": self._name,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]
        if stream:
            payload["stream"] = True
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._payload(request, stream=False)

        async def _call() -> dict[str, Any]:
            return await self._client.post_json("/messages", payload)

        data = await with_retry(_call)
        blocks = data.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tool_calls = [
            ToolCall(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input", {}) or {})
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        usage_raw = data.get("usage", {}) or {}
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
        )
        usage = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self.info.cost.estimate(usage),
        )
        stop = data.get("stop_reason") or "end_turn"
        finish = "tool_calls" if tool_calls else ("length" if stop == "max_tokens" else "stop")
        return GenerationResult(
            content=[TextContent(text=text)],
            tool_calls=tool_calls,
            finish_reason=finish,  # type: ignore[arg-type]
            model=self.info.ref,
            usage=usage,
            raw={"provider_response_id": data.get("id")},
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        payload = self._payload(request, stream=True)
        async for event in self._client.stream_sse("/messages", payload):
            etype = event.get("type")
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield GenerationChunk(text=delta.get("text", ""))
            elif etype == "message_stop":
                yield GenerationChunk(finish_reason="stop")

    async def health(self) -> HealthStatus:
        try:
            await self.generate(GenerationRequest.of("ping", max_tokens=1))
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()


def register(runtime: Runtime) -> None:
    def _factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        cred = runtime.settings.credential("anthropic")
        api_key = options.get("api_key") or cred.resolve_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "anthropic",
                "no API key: set ANTHROPIC_API_KEY, providers.anthropic.api_key, or pass api_key=",
            )
        base_url = options.get("base_url") or cred.base_url or DEFAULT_BASE_URL
        client = ProviderHttpClient(
            runtime,
            "anthropic",
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
                **cred.default_headers,
            },
        )
        return AnthropicModel(name, client)

    runtime.model_providers.register("anthropic", _factory, replace=True)
