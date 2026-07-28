"""0.2.3 hardening tests: cache correctness, trainer resume, OTLP exporter."""

from __future__ import annotations

import pytest

from aire.core.content import TextContent
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.models.builtin import CallableModel, HashingEmbedder
from aire.models.types import GenerationRequest, StructuredOutputSpec
from aire.observability.otlp import OTLPExporter
from aire.observability.tracing import SpanRecord, Tracer
from aire.optimization.cache import CachedModel, SemanticCachedModel
from aire.training.trainer import FunctionTrainer, TrainingConfig
from tests.conftest import arun


def _request(prompt: str, **kwargs) -> GenerationRequest:
    return GenerationRequest.of(prompt, **kwargs)


# -- cache correctness -----------------------------------------------------------------


def _text_model(text: str) -> CallableModel:
    return CallableModel("t", lambda prompt: text)


def test_exact_cache_returns_independent_copies() -> None:
    cached = CachedModel(_text_model("hello"))
    first = arun(cached.generate(_request("q")))
    first.content.append(TextContent(text="MUTATED"))  # caller mutates its copy
    second = arun(cached.generate(_request("q")))
    assert second.text == "hello"  # mutation never leaks into the cache
    assert len(second.content) == 1


def test_semantic_cache_respects_generation_params() -> None:
    cached = SemanticCachedModel(_text_model("answer"), HashingEmbedder("h"), threshold=0.9)
    plain = arun(cached.generate(_request("what is aire")))
    assert plain.text == "answer"
    assert cached.stats()["misses"] == 1

    # Same prompt, different temperature → must NOT hit
    arun(cached.generate(_request("what is aire", temperature=0.9)))
    assert cached.stats()["hits"] == 0
    assert cached.stats()["misses"] == 2

    # Same prompt + same params → hits
    arun(cached.generate(_request("what is aire", temperature=0.9)))
    assert cached.stats()["hits"] == 1

    # Structured-output request must not be served the plain-text entry
    spec = StructuredOutputSpec(
        name="S", schema={"type": "object", "properties": {"x": {"type": "string"}}}
    )
    arun(cached.generate(_request("what is aire", response_format=spec)))
    assert cached.stats()["misses"] == 3


def test_semantic_cache_returns_independent_copies() -> None:
    cached = SemanticCachedModel(_text_model("fixed"), HashingEmbedder("h"))
    first = arun(cached.generate(_request("q")))
    first.content.append(TextContent(text="MUTATED"))
    second = arun(cached.generate(_request("q")))
    assert cached.stats()["hits"] == 1
    assert second.text == "fixed"
    assert len(second.content) == 1


# -- trainer resume -------------------------------------------------------------------


def _counting_step(calls: list[int]):
    def step(epoch, dataset, config, state):
        calls.append(epoch)
        state["seen"] = state.get("seen", 0) + len(dataset.records)
        return {"loss": 1.0 / (epoch + 1)}, state

    return step


def test_trainer_resume_continues_from_checkpoint(tmp_path) -> None:
    dataset = Dataset([Record(text="a"), Record(text="b")], name="t")
    config = TrainingConfig(epochs=4, checkpoint_dir=str(tmp_path))

    calls: list[int] = []
    trainer = FunctionTrainer(_counting_step(calls), config)
    # Simulate an interrupted run: only 2 epochs, then "crash"
    trainer.config.epochs = 2
    first = arun(trainer.fit(dataset))
    assert calls == [0, 1]
    checkpoint_path = first.checkpoints[-1].path
    assert checkpoint_path is not None

    # Resume: epochs 2..3 run, state carries over
    trainer.config.epochs = 4
    resumed = arun(trainer.fit(dataset, resume_from=checkpoint_path))
    assert calls == [0, 1, 2, 3]
    assert resumed.checkpoints[-1].state["seen"] == 8  # 2 records x 4 epochs
    assert resumed.best_metric == pytest.approx(0.25)


def test_trainer_resume_exhausted_raises(tmp_path) -> None:
    dataset = Dataset([Record(text="a")], name="t")
    config = TrainingConfig(epochs=2, checkpoint_dir=str(tmp_path))
    trainer = FunctionTrainer(_counting_step([]), config)
    result = arun(trainer.fit(dataset))
    from aire.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        arun(trainer.fit(dataset, resume_from=result.checkpoints[-1].path))


def test_load_checkpoint_missing_raises(tmp_path) -> None:
    from aire.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        FunctionTrainer.load_checkpoint(tmp_path / "nope.json")


# -- OTLP exporter ---------------------------------------------------------------------


def _span_record(name: str = "op", *, error: str | None = None) -> SpanRecord:
    return SpanRecord(
        trace_id="ab" * 16,
        span_id="cd" * 8,
        parent_span_id="ef" * 8,
        name=name,
        start_time=1_700_000_000.0,
        end_time=1_700_000_000.25,
        attributes={"model": "mock:default", "tokens": 42, "ok": True},
        status="error" if error else "ok",
        error=error,
    )


def test_otlp_payload_shape() -> None:
    exporter = OTLPExporter("http://collector:4318", service_name="svc")
    payload = exporter.payload([_span_record(error="boom")])
    resource_spans = payload["resourceSpans"][0]
    assert resource_spans["resource"]["attributes"][0]["value"]["stringValue"] == "svc"
    span = resource_spans["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "ab" * 16
    assert span["parentSpanId"] == "ef" * 8
    assert span["status"]["code"] == 2
    assert span["events"][0]["name"] == "exception"
    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert attrs["tokens"] == {"intValue": "42"}
    assert attrs["ok"] == {"boolValue": True}
    assert attrs["model"] == {"stringValue": "mock:default"}


def test_otlp_flush_posts_and_counts() -> None:
    import json

    import httpx

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exporter = OTLPExporter("http://collector:4318", batch_size=2, client=client)
    tracer = Tracer(exporter=exporter)
    with tracer.span("one"):
        pass
    assert not seen  # batched, not yet flushed
    with tracer.span("two"):
        pass
    assert len(seen) == 1  # auto-flushed at batch_size
    assert len(seen[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2
    assert exporter.exported == 2
    assert exporter.failures == 0


def test_otlp_failures_never_raise() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exporter = OTLPExporter("http://collector:4318", client=client)
    exporter.export(_span_record())
    exporter.flush()  # must not raise
    assert exporter.failures == 1
    assert "500" in (exporter.last_error or "")


def test_tracer_mask_fields_case_insensitive() -> None:
    exporter_records: list[SpanRecord] = []

    class Capture:
        def export(self, record: SpanRecord) -> None:
            exporter_records.append(record)

    tracer = Tracer(exporter=Capture(), mask_fields=["API_Key"])
    with tracer.span("s", attributes={"api_key": "secret", "other": "fine"}):
        pass
    assert exporter_records[0].attributes == {"api_key": "***", "other": "fine"}
