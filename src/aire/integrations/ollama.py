"""Ollama local inference provider (``"ollama:<model>"``).

Ollama serves an OpenAI-compatible API plus a native API; this adapter uses the
native ``/api/chat`` and ``/api/embed`` endpoints. No credentials required —
perfect for private, offline-capable development.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from aire.core.content import TextContent
from aire.core.types import Capability, HealthStatus, Usage
from aire.integrations.http import ProviderHttpClient
from aire.models.base import EmbeddingModel, Model, estimate_tokens
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

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaModel(Model):
    def __init__(self, name: str, client: ProviderHttpClient) -> None:
        self._name = name
        self._client = client

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            ref=f"ollama:{self._name}",
            provider="ollama",
            capabilities=[
                Capability.TEXT_GENERATION,
                Capability.STREAMING,
                Capability.TOOL_CALLING,
            ],
            hardware="local",
        )

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._name,
            "messages": [{"role": m.role, "content": m.text_content} for m in request.messages],
            "options": {},
        }
        options: dict[str, Any] = payload["options"]
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        if request.seed is not None:
            options["seed"] = request.seed
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
        if request.response_format is not None:
            payload["format"] = request.response_format.json_schema
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {**self._payload(request), "stream": False}
        data = await self._client.post_json("/api/chat", payload)
        message = data.get("message", {}) or {}
        tool_calls = [
            ToolCall(
                id=f"ollama-{i}",
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", {}) or {},
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        usage = Usage(
            input_tokens=data.get("prompt_eval_count")
            or estimate_tokens("\n".join(m.text_content for m in request.messages)),
            output_tokens=data.get("eval_count", 0),
        )
        return GenerationResult(
            content=[TextContent(text=message.get("content", ""))],
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            model=self.info.ref,
            usage=usage,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        payload = {**self._payload(request), "stream": True}
        # Ollama streams newline-delimited JSON rather than SSE.
        async with self._client.raw.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message", {}) or {}
                text = message.get("content", "")
                done = data.get("done", False)
                yield GenerationChunk(text=text, finish_reason="stop" if done else None)

    async def health(self) -> HealthStatus:
        try:
            await self._client.raw.get("/api/tags")
        except Exception as exc:
            return HealthStatus.unhealthy(f"ollama unreachable (is `ollama serve` running?): {exc}")
        return HealthStatus.healthy()


class OllamaEmbedder(EmbeddingModel):
    def __init__(self, name: str, client: ProviderHttpClient) -> None:
        self._name = name
        self._client = client
        self._dimension = 0

    @property
    def name(self) -> str:
        return f"ollama:{self._name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        data = await self._client.post_json(
            "/api/embed", {"model": self._name, "input": request.inputs}
        )
        vectors = data.get("embeddings", []) or []
        if vectors and not self._dimension:
            self._dimension = len(vectors[0])
        return EmbeddingResult(vectors=vectors, model=self.name)


def register(runtime: Runtime) -> None:
    def _client(runtime: Runtime, options: dict[str, Any]) -> ProviderHttpClient:
        import os

        base_url = (
            options.get("base_url")
            or runtime.settings.credential("ollama").base_url
            or os.environ.get("OLLAMA_HOST")
            or DEFAULT_BASE_URL
        )
        return ProviderHttpClient(runtime, "ollama", base_url=base_url)

    def _model_factory(name: str, *, runtime: Runtime, **options: Any) -> Model:
        return OllamaModel(name, _client(runtime, options))

    def _embedder_factory(name: str, *, runtime: Runtime, **options: Any) -> EmbeddingModel:
        return OllamaEmbedder(name, _client(runtime, options))

    runtime.model_providers.register("ollama", _model_factory, replace=True)
    runtime.embedders.register("ollama", _embedder_factory, replace=True)
