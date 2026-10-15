"""Offline audit probes; prints observed behavior, not a passing test assertion.

Run with the development environment from any working directory.
All transports/models are local fakes; no external provider/service is contacted.
Findings reflect the currently installed aire source, so change as fixes land.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from aire.agents.runtime import AgentExecutor
from aire.agents.streaming import stream_tokens
from aire.agents.types import AgentConfig
from aire.core.config import Settings
from aire.core.content import Message
from aire.core.context import ExecutionContext
from aire.core.runtime import Runtime
from aire.deployment.artifacts import generate_artifacts
from aire.deployment.gateway import create_gateway
from aire.integrations.qdrant import QdrantVectorStore, _to_scored
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.models.types import GenerationRequest, ToolCall
from aire.optimization.cache import CachedModel, SemanticCachedModel
from aire.rag.types import Chunk
from aire.safety.guardrails import GuardrailVerdict
from aire.tools.registry import ToolRegistry
from aire.tools.tool import Tool


class OutputModel(EchoModel):
    async def generate(self, request):
        from aire.models.types import GenerationResult

        return GenerationResult.text_result("BLOCKME", model=self.info.ref)

    async def stream(self, request):
        from aire.models.types import GenerationChunk

        yield GenerationChunk(text="BLOCKME", finish_reason="stop")


class BlockOutput:
    name = "audit"

    def check(self, text):
        return GuardrailVerdict(passed="BLOCKME" not in text, guardrail=self.name)


class FakeRaw:
    def __init__(self):
        self.writes = []

    async def put(self, path, json):
        self.writes.append((path, json))
        return type("Response", (), {"status_code": 200})()

    async def post(self, path, json):
        self.writes.append((path, json))
        return type("Response", (), {"status_code": 500})()


class FakeClient:
    def __init__(self):
        self.raw = FakeRaw()

    async def get_json(self, path):
        return {}


async def main():
    findings = {}
    with tempfile.TemporaryDirectory() as d:
        generate_artifacts(d)
        findings["deployment_requirements"] = (
            Path(d, "requirements.in").read_text().strip()
        )

    def slow():
        time.sleep(0.04)
        return "finished"

    started = time.monotonic()
    result = await Tool(slow, timeout_seconds=0.001).execute()
    findings["sync_tool_timeout"] = {
        "ok": result.ok,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    ctx = ExecutionContext()
    ctx.cancel()
    result = await AgentExecutor(EchoModel(), ToolRegistry()).run(
        "cancelled", context=ctx
    )
    findings["cancelled_agent"] = str(result.status)
    executor = AgentExecutor(
        EchoModel(), ToolRegistry(), config=AgentConfig(token_budget=0)
    )
    findings["agent_zero_budget"] = str((await executor.run("hello")).status)
    findings["stream_zero_budget"] = "".join(
        [x async for x in stream_tokens(executor, "hello")]
    )
    inner = EchoModel()
    inner.scripted_tool_calls = [ToolCall(id="t", name="test", arguments={"a": 1})]
    cache = CachedModel(inner)
    request = GenerationRequest.of("hello")
    await cache.generate(request)
    hit = [x async for x in cache.stream(request)][0]
    hit.tool_calls[0].arguments["a"] = 999999
    findings["stream_cache_mutation"] = (
        (await cache.generate(request)).tool_calls[0].arguments["a"]
    )
    sem = SemanticCachedModel(EchoModel(), HashingEmbedder())
    await sem.generate(
        GenerationRequest(
            messages=[Message.text("system", "alpha"), Message.text("user", "beta")]
        )
    )
    await sem.generate(
        GenerationRequest(
            messages=[Message.text("user", "alpha"), Message.text("system", "beta")]
        )
    )
    findings["semantic_role_collision_hits"] = sem.hits
    client = FakeClient()
    store = QdrantVectorStore("audit", client)
    await store.upsert(
        [Chunk(id="stable", document_id="doc-1", text="old", embedding=[1.0])]
    )
    findings["qdrant_upsert"] = client.raw.writes[-1][1]
    findings["qdrant_delete_500_return"] = await store.delete(["stable"])
    findings["qdrant_delete_request"] = client.raw.writes[-1][1]
    findings["qdrant_document_roundtrip"] = _to_scored(
        {"id": 1, "payload": findings["qdrant_upsert"]["points"][0]["payload"]}
    ).chunk.document_id
    async with Runtime(Settings()) as runtime:
        runtime.model_providers.register("audit", lambda name, **kw: OutputModel())
        app = create_gateway(
            runtime, models=["audit:output"], guardrails=[BlockOutput()]
        )
        with TestClient(app) as web:
            body = {
                "model": "audit:output",
                "messages": [{"role": "user", "content": "hello"}],
            }
            a = web.post("/v1/chat/completions", json=body)
            b = web.post("/v1/chat/completions", json={**body, "stream": True})
            findings["gateway_output_guard"] = {
                "normal_status": a.status_code,
                "stream_status": b.status_code,
                "stream_leaks_blocked_text": "BLOCKME" in b.text,
            }
    print(json.dumps(findings, indent=2))


asyncio.run(main())
