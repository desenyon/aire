"""Anthropic Messages API provider (``"anthropic:<model>"``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from aire.core.content import TextContent
from aire.core.errors import AuthenticationError
from aire.core.plugins import PluginInfo
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
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "",
                                "content": m.text_content,
                            }
                        ],
                    }
                )
                continue
            content = self._anthropic_content(m)
            messages.append(
                {
                    "role": m.role if m.role in {"user", "assistant"} else "user",
                    "content": content,
                }
            )
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
        if request.tool_choice is not None:
            payload["tool_choice"] = self._tool_choice(request.tool_choice)
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _tool_choice(choice: str) -> dict[str, Any]:
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": choice}

    @staticmethod
    def _anthropic_content(message: Any) -> Any:
        from aire.core.content import ImageContent, StructuredContent, TextContent

        blocks: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextContent) and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.data is not None:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type or "image/png",
                                "data": part.as_base64(),
                            },
                        }
                    )
                elif part.uri:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.uri},
                        }
                    )
            elif (
                isinstance(part, StructuredContent)
                and isinstance(part.data, dict)
                and part.data.get("type") == "tool_use"
            ):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.data.get("id") or "",
                        "name": part.data.get("name") or "",
                        "input": part.data.get("input") or {},
                    }
                )
        existing = {b.get("id") for b in blocks if b.get("type") == "tool_use"}
        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in message.tool_calls
            if c.id not in existing
        )
        return blocks if blocks else message.text_content

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

    async def stream(  # noqa: C901 — explicit provider event state machine
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        payload = self._payload(request, stream=True)
        from aire.core.errors import ProviderError
        from aire.models.streaming import ToolCallBuffer

        pending: dict[int, ToolCallBuffer] = {}
        input_tokens = output_tokens = 0
        finish = "stop"
        async for event in self._client.stream_sse("/messages", payload):
            etype = event.get("type")
            index = event.get("index", 0)
            if etype == "message_start":
                usage = event.get("message", {}).get("usage") or {}
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            elif etype == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    pending[index] = ToolCallBuffer(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        initial=block.get("input") or {},
                    )
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield GenerationChunk(text=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    if index not in pending:
                        raise ProviderError(
                            "anthropic",
                            "tool delta has no matching block",
                            code="provider.stream_tool_invalid",
                            retryable=False,
                        )
                    pending[index].fragments.append(delta.get("partial_json", ""))
            elif etype == "content_block_stop" and index in pending:
                yield GenerationChunk(tool_calls=[pending.pop(index).finish("anthropic")])
            elif etype == "message_delta":
                reason = event.get("delta", {}).get("stop_reason")
                finish = {"tool_use": "tool_calls", "max_tokens": "length"}.get(reason, "stop")
                output_tokens = (event.get("usage") or {}).get("output_tokens", output_tokens)
            elif etype == "message_stop":
                if pending:
                    raise ProviderError(
                        "anthropic",
                        "stream ended before tool calls completed",
                        code="provider.stream_incomplete",
                        retryable=False,
                    )
                tokens = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
                yield GenerationChunk(
                    finish_reason=finish,
                    usage=tokens.model_copy(update={"cost_usd": self.info.cost.estimate(tokens)}),
                )
            elif etype == "error":
                raise ProviderError(
                    "anthropic", "provider reported a stream error", code="provider.stream_error"
                )
        if pending:
            raise ProviderError(
                "anthropic",
                "stream ended before tool calls completed",
                code="provider.stream_incomplete",
                retryable=False,
            )

    async def health(self) -> HealthStatus:
        try:
            await self.generate(GenerationRequest.of("ping", max_tokens=1))
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()


def register(runtime: Runtime) -> PluginInfo:
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
    return PluginInfo(name="anthropic", version="0.1.0", provides=["model:anthropic"])


class AnthropicProvider:
    """Entry-point target for the ``anthropic`` provider."""

    @staticmethod
    def register(runtime: Runtime) -> PluginInfo:
        return register(runtime)
