import pytest

from aire.core.content import ImageContent, Message, TextContent
from aire.models.base import run_sync
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.models.types import GenerationRequest, ToolCall
from aire.optimization.cache import CachedModel, SemanticCachedModel


def test_stream_hit_does_not_expose_cached_arguments():
    async def check():
        model = EchoModel()
        model.scripted_tool_calls = [
            ToolCall(id="one", name="read", arguments={"nested": {"x": 1}})
        ]
        cache = CachedModel(model)
        request = GenerationRequest.of("hello")
        await cache.generate(request)
        chunks = [chunk async for chunk in cache.stream(request)]
        hit = chunks[0]
        hit.tool_calls[0].arguments["nested"]["x"] = 2
        assert (await cache.generate(request)).tool_calls[0].arguments["nested"]["x"] == 1

    run_sync(check())


@pytest.mark.parametrize("change", ["role", "system", "media", "namespace"])
def test_semantic_cache_keeps_context_boundaries(change):
    async def check():
        cache = SemanticCachedModel(EchoModel(), HashingEmbedder(), threshold=0)
        a = GenerationRequest(
            messages=[Message.text("system", "alpha"), Message.text("user", "beta")]
        )
        if change == "role":
            b = GenerationRequest(
                messages=[Message.text("user", "alpha"), Message.text("system", "beta")]
            )
        elif change == "system":
            b = GenerationRequest(
                messages=[Message.text("system", "other"), Message.text("user", "beta")]
            )
        elif change == "media":
            b = GenerationRequest(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextContent(text="beta"),
                            ImageContent(uri="https://example.com/image"),
                        ],
                    )
                ]
            )
        else:
            b = a.model_copy(update={"metadata": {"cache_namespace": "other"}})
        await cache.generate(a)
        await cache.generate(b)
        assert cache.hits == 0

    run_sync(check())


@pytest.mark.parametrize("change", ["role", "system", "media", "namespace"])
def test_gateway_semantic_cache_keeps_context_boundaries(change):
    from aire.core.config import Settings
    from aire.core.runtime import Runtime
    from aire.deployment.gateway import Gateway

    async def check():
        async with Runtime(Settings()) as runtime:
            gateway = Gateway(
                runtime,
                chat_routes={"echo": ["mock:echo"]},
                semantic_cache=True,
                semantic_threshold=0,
            )
            a = GenerationRequest(
                messages=[Message.text("system", "alpha"), Message.text("user", "beta")]
            )
            if change == "role":
                b = GenerationRequest(
                    messages=[Message.text("user", "alpha"), Message.text("system", "beta")]
                )
            elif change == "system":
                b = GenerationRequest(
                    messages=[Message.text("system", "other"), Message.text("user", "beta")]
                )
            elif change == "media":
                b = GenerationRequest(
                    messages=[
                        Message(
                            role="user",
                            content=[
                                TextContent(text="beta"),
                                ImageContent(uri="https://example.com/image"),
                            ],
                        )
                    ]
                )
            else:
                b = a.model_copy(update={"metadata": {"cache_namespace": "other"}})
            await gateway.generate("echo", a)
            await gateway.generate("echo", b)
            assert gateway.semantic_hits == 0

    run_sync(check())
