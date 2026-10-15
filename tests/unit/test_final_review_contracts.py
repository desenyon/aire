import asyncio
import json

import pytest
from pydantic import ValidationError

from aire.core.config import Settings
from aire.core.context import Budget, ExecutionContext
from aire.core.runtime import Runtime
from aire.deployment.gateway import Gateway
from aire.models.base import run_sync
from aire.models.builtin import HashingEmbedder
from aire.models.types import GenerationRequest, GenerationResult
from aire.tools.schema import schema_model
from aire.tools.tool import Tool
from aire.tools.types import RetryPolicy, SideEffect


@pytest.mark.parametrize("stop", ["attempt", "backoff", "deadline", "already_cancelled"])
def test_tool_context_stops_further_effects(stop):
    async def check():
        ctx = ExecutionContext(budget=Budget(deadline_seconds=0.01 if stop == "deadline" else None))
        calls = []
        failed = asyncio.Event()

        async def write():
            calls.append("effect")
            if stop == "attempt":
                ctx.cancel()
            failed.set()
            raise RuntimeError("connection lost after write")

        async def cancel_during_backoff():
            await failed.wait()
            await asyncio.sleep(0)
            ctx.cancel()

        cancellation = None
        if stop == "already_cancelled":
            ctx.cancel()
        elif stop == "backoff":
            cancellation = asyncio.create_task(cancel_during_backoff())
        result = await Tool(
            write,
            retry=RetryPolicy(attempts=3, backoff_seconds=0.02),
            side_effect=SideEffect.EXTERNAL_SIDE_EFFECT,
        ).execute(context=ctx)
        if cancellation:
            await cancellation
        assert calls == ([] if stop == "already_cancelled" else ["effect"])
        assert not result.ok
        assert result.error_code == ("aire.timeout" if stop == "deadline" else "aire.cancelled")

    run_sync(check())


@pytest.mark.parametrize(
    ("prop", "invalid", "valid"),
    [
        ({"type": "array", "items": {"type": "integer", "minimum": 1}}, [-9], [1]),
        ({"anyOf": [{"type": "string", "minLength": 4}, {"type": "null"}]}, "x", None),
        ({"type": "array", "items": {"type": "string", "pattern": "^ok"}}, ["bad"], ["okay"]),
        ({"type": "array", "items": {"type": "array", "minItems": 2}}, [[1]], [[1, 2]]),
        ({"type": ["string", "null"], "minLength": 4}, "x", None),
        ({"type": "number", "nullable": True, "exclusiveMinimum": 2}, 1, None),
    ],
)
def test_imported_constraints_apply_recursively(prop, invalid, valid):
    model = schema_model(
        "Args", {"type": "object", "properties": {"value": prop}, "required": ["value"]}
    )
    with pytest.raises(ValidationError):
        model.model_validate({"value": invalid})
    assert model.model_validate({"value": valid}).model_dump()["value"] == valid


class SharedCache:
    def __init__(self):
        self.values = {}

    async def aget(self, key):
        return self.values.get(key)

    async def aset(self, key, value, **kwargs):
        self.values[key] = value


@pytest.mark.parametrize("change", ["route", "order", "embedder", "routing", "namespace"])
def test_persistent_gateway_cache_separates_configuration(change):
    async def check():
        async with Runtime(Settings()) as runtime:
            backend = SharedCache()
            options = dict(
                chat_routes={"chat": ["mock:old", "mock:new"]},
                semantic_cache=True,
                semantic_embedder=HashingEmbedder(),
                semantic_cache_redis_url="redis://shared",
            )
            first = Gateway(runtime, **options)
            first._semantic_redis_backend = backend
            request = GenerationRequest.of("hello")
            await first._semantic_store(
                "chat",
                request,
                GenerationResult.text_result("OLD ANSWER", model="mock:old"),
                resolved="mock:old",
            )
            same = Gateway(runtime, **options)
            same._semantic_redis_backend = backend
            assert (await same.generate("chat", request))[1].text == "OLD ANSWER"
            if change == "route":
                options["chat_routes"] = {"chat": ["mock:new"]}
            elif change == "order":
                options["chat_routes"] = {"chat": ["mock:new", "mock:old"]}
            elif change == "embedder":
                options["semantic_embedder"] = HashingEmbedder(name="other-space")
            elif change == "routing":
                options["routing"] = "round_robin"
            else:
                options["semantic_cache_namespace"] = "new-deployment"
            replacement = Gateway(runtime, **options)
            replacement._semantic_redis_backend = backend
            assert (await replacement.generate("chat", request))[1].text != "OLD ANSWER"
            assert replacement.semantic_hits == 0

    run_sync(check())


def test_gateway_rejects_cache_entry_outside_current_route():
    async def check():
        async with Runtime(Settings()) as runtime:
            gateway = Gateway(
                runtime,
                chat_routes={"chat": ["mock:new"]},
                semantic_cache=True,
                semantic_embedder=HashingEmbedder(),
                semantic_cache_redis_url="redis://shared",
            )
            backend = SharedCache()
            gateway._semantic_redis_backend = backend
            request = GenerationRequest.of("hello")
            await gateway._semantic_store(
                "chat",
                request,
                GenerationResult.text_result("new", model="mock:new"),
                resolved="mock:new",
            )
            key = next(iter(backend.values))
            payload = json.loads(backend.values[key])
            payload[0]["ref"] = "mock:removed"
            payload[0]["result"] = GenerationResult.text_result(
                "OLD ANSWER", model="mock:removed"
            ).model_dump(mode="json")
            backend.values[key] = json.dumps(payload)
            assert (await gateway.generate("chat", request))[1].text != "OLD ANSWER"

    run_sync(check())
