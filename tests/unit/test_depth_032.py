"""0.3.2 depth: eval metrics, rerankers, cost policy, quant/distill, vision/audio."""

from __future__ import annotations

import pytest

from aire.agents.memory import resolve_memory
from aire.audio import AudioPipeline
from aire.core.content import ImageContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.evaluation import EvalCase, Evaluator
from aire.evaluation.metrics import get_metric, metric_names
from aire.memory import LongTermMemory
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.models.types import CostInfo, GenerationRequest, ModelInfo
from aire.optimization import CostPolicy, ModelRouter
from aire.rag import Chunk, EmbeddingReranker, ModelReranker, ScoredChunk, get_reranker
from aire.training import Distiller, Quantizer, create_distiller, create_quantizer, soft_kl_loss
from aire.vision import ImageGenerationPipeline, VisionPipeline
from tests.conftest import arun


class _Ctx:
    latency_ms = 1.0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    context: str | None = None
    judge = None
    embedder = None


def test_new_metrics_registered() -> None:
    for name in ("faithfulness", "embedding_similarity", "bleu", "rouge_l"):
        assert name in metric_names()


def test_faithfulness_and_bleu_rouge() -> None:
    case = EvalCase(
        input="q",
        expected="refunds are allowed within thirty days",
        context="refunds are allowed within thirty days for unused items",
    )
    ctx = _Ctx()
    ctx.context = case.context
    faith = arun(get_metric("faithfulness")(case, "refunds are allowed within thirty days", ctx))
    assert faith.score > 0.5
    bleu = arun(get_metric("bleu")(case, "refunds are allowed within thirty days", ctx))
    assert bleu.score > 0.5
    rouge = arun(get_metric("rouge_l")(case, "refunds are allowed within thirty days", ctx))
    assert rouge.score > 0.5


def test_embedding_similarity_with_embedder() -> None:
    case = EvalCase(input="q", expected="hello world")
    ctx = _Ctx()
    ctx.embedder = HashingEmbedder()
    same = arun(get_metric("embedding_similarity")(case, "hello world", ctx))
    different = arun(get_metric("embedding_similarity")(case, "zzzz unrelated", ctx))
    assert same.score >= different.score


def test_evaluator_passes_embedder() -> None:
    report = arun(
        Evaluator(embedder=HashingEmbedder()).run(
            EchoModel(),
            [EvalCase(input="ping", expected="ping")],
            metrics=["embedding_similarity"],
        )
    )
    assert report.results[0].metrics[0].name == "embedding_similarity"
    assert report.results[0].metrics[0].score > 0


def test_embedding_and_model_rerankers() -> None:
    embedder = HashingEmbedder()
    chunks = [
        Chunk(text="refund policy thirty days", embedding=arun(embedder.embed_one("refund"))),
        Chunk(text="oauth tokens rotate", embedding=arun(embedder.embed_one("oauth"))),
    ]
    hits = [ScoredChunk(chunk=c, score=0.1) for c in chunks]
    emb = EmbeddingReranker(embedder)
    ranked = arun(emb.rerank("refund window", hits, k=2))
    assert "refund" in ranked[0].chunk.text
    model_rr = ModelReranker(EchoModel())
    ranked2 = arun(model_rr.rerank("refund", hits, k=1))
    assert len(ranked2) == 1
    assert get_reranker("lexical").__class__.__name__ == "LexicalOverlapReranker"


def test_cost_policy_blocks_expensive() -> None:
    class Priced(EchoModel):
        def __init__(self, name: str, price: float) -> None:
            super().__init__(name)
            self._price = price

        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(
                update={
                    "ref": f"mock:{self._name}",
                    "cost": CostInfo(input_per_million=self._price, output_per_million=self._price),
                }
            )

    policy = CostPolicy(max_cost_per_request_usd=0.000001, prefer_cheaper_within=1.0)
    cheap = Priced("cheap", 0.01)
    pricey = Priced("pricey", 1_000_000.0)
    router = ModelRouter([pricey, cheap], objective="highest_quality", policy=policy)
    decision = router.route(GenerationRequest.of("hello"))
    assert decision.chosen == "mock:cheap"
    assert "mock:pricey" in decision.policy_blocked
    arun(router.generate(GenerationRequest.of("hello")))
    assert policy.requests_today == 1


def test_quantizer_stub_and_distiller() -> None:
    q = create_quantizer("gpt2", method="stub", bits=4)
    assert q.prepare()["bits"] == 4
    assert q.describe()["kind"] == "quantizer"
    soft = soft_kl_loss([1.0, 0.0], [0.9, 0.1], temperature=2.0)
    assert soft >= 0.0
    d = create_distiller(temperature=2.0, alpha=0.7)
    result = d.step([1.0, 0.0], [0.8, 0.2])
    assert result.total_loss >= 0.0
    assert Distiller().describe()["kind"] == "distiller"
    assert Quantizer("x", config=q.config).describe()["available"]["stub"] is True


def test_vision_detect_and_image_gen() -> None:
    class VisionEcho(EchoModel):
        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(
                update={
                    "capabilities": [
                        *super().info.capabilities,
                        Capability.VISION_INPUT,
                        Capability.IMAGE_GENERATION,
                    ]
                }
            )

        async def generate(self, request):  # type: ignore[override]
            from aire.models.types import GenerationResult

            text = request.messages[0].text_content if request.messages else ""
            if "Detect objects" in text:
                return GenerationResult.text_result(
                    '{"detections":[{"label":"cat","confidence":0.9,"box":[0,0,1,1]}]}',
                    model=self.info.ref,
                )
            if "Generate an image" in text:
                return GenerationResult.text_result(
                    "https://example.com/gen.png",
                    model=self.info.ref,
                )
            return await super().generate(request)

    model = VisionEcho()
    pipeline = VisionPipeline(model)
    result = arun(pipeline.detect(ImageContent(uri="https://example.com/x.png"), labels=["cat"]))
    assert result.detections and result.detections[0].label == "cat"
    gen = ImageGenerationPipeline(model)
    image = arun(gen.generate("a red cube"))
    assert image.uri == "https://example.com/gen.png"


def test_audio_synthesize() -> None:
    class TtsEcho(EchoModel):
        @property
        def info(self) -> ModelInfo:
            return super().info.model_copy(
                update={"capabilities": [*super().info.capabilities, Capability.TEXT_TO_SPEECH]}
            )

        async def generate(self, request):  # type: ignore[override]
            from aire.models.types import GenerationResult

            return GenerationResult.text_result("file:/tmp/speech.wav", model=self.info.ref)

    pipe = AudioPipeline(TtsEcho())
    out = arun(pipe.synthesize("hello"))
    assert out.audio_uri == "file:/tmp/speech.wav"
    with pytest.raises(NotFoundError):
        arun(AudioPipeline(EchoModel()).synthesize("x"))


def test_resolve_memory_long_term(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mem = resolve_memory(f"long-term:{tmp_path / 'mem'}")
    assert isinstance(mem, LongTermMemory)
    assert resolve_memory("long-term").__class__.__name__ == "LongTermMemory"
