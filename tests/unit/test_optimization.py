"""Optimization: response caches and the model router."""

from __future__ import annotations

import pytest

from aire.core.errors import ContextLengthError, NotFoundError
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.models.types import GenerationRequest, ModelInfo
from aire.optimization import CachedModel, ModelRouter, SemanticCachedModel, assert_fits
from tests.conftest import arun


def test_cached_model_hits() -> None:
    cached = CachedModel(EchoModel())
    request = GenerationRequest.of("repeat me")
    first = arun(cached.generate(request))
    second = arun(cached.generate(request))
    assert first.text == second.text
    stats = cached.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_cached_model_respects_params() -> None:
    cached = CachedModel(EchoModel())
    a = arun(cached.generate(GenerationRequest.of("x", temperature=0.1)))
    b = arun(cached.generate(GenerationRequest.of("x", temperature=0.9)))
    assert cached.stats()["misses"] == 2
    assert a.text == b.text  # echo model; distinct cache entries though


def test_semantic_cache_similar_prompt() -> None:
    semantic = SemanticCachedModel(EchoModel(), HashingEmbedder(), threshold=0.99)
    arun(semantic.generate(GenerationRequest.of("what is the refund policy")))
    arun(semantic.generate(GenerationRequest.of("what is the refund policy")))
    assert semantic.stats()["hits"] == 1


def test_router_picks_cheapest() -> None:
    from aire.models.types import CostInfo

    class PricedEcho(EchoModel):
        def __init__(self, name: str, price: float) -> None:
            super().__init__(name)
            self._price = price

        @property
        def info(self) -> ModelInfo:
            base = super().info
            return base.model_copy(
                update={
                    "ref": f"mock:{self._name}",
                    "cost": CostInfo(input_per_million=self._price, output_per_million=self._price),
                }
            )

    cheap = PricedEcho("cheap", 0.01)
    pricey = PricedEcho("pricey", 100.0)
    router = ModelRouter([pricey, cheap], objective="lowest_cost")
    decision = router.route(GenerationRequest.of("hello"))
    assert decision.chosen == "mock:cheap"
    result = arun(router.generate(GenerationRequest.of("hello")))
    assert result.text == "hello"


def test_router_requires_tool_capability() -> None:
    from aire.core.types import Capability
    from aire.models.types import ToolDefinition

    class NoTools(EchoModel):
        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(update={"capabilities": [Capability.TEXT_GENERATION]})

    router = ModelRouter([NoTools("notools")])
    request = GenerationRequest.of("hi", tools=[ToolDefinition(name="t")])
    with pytest.raises(NotFoundError):
        router.route(request)


def test_router_fallback_on_failure() -> None:
    class Failing(EchoModel):
        async def generate(self, request):  # type: ignore[override]
            raise RuntimeError("down")

    router = ModelRouter([Failing("bad"), EchoModel("good")], objective="highest_quality")
    result = arun(router.generate(GenerationRequest.of("ping")))
    assert result.text == "ping"
    assert router.history["mock:bad"].success_rate == 0.0
    assert router.history["mock:good"].calls == 1


def test_router_context_window_exclusion() -> None:
    class Tiny(EchoModel):
        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(update={"context_window": 2})

    router = ModelRouter([Tiny("tiny")])
    with pytest.raises(NotFoundError):
        router.route(GenerationRequest.of("a very long prompt that will not fit"))


def test_assert_fits() -> None:
    class Tiny(EchoModel):
        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(update={"context_window": 2})

    with pytest.raises(ContextLengthError):
        assert_fits(Tiny(), GenerationRequest.of("way too much text for the window"))
    assert_fits(EchoModel(), GenerationRequest.of("fine"))
