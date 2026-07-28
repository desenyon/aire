"""Tracing, metrics and event observation."""

from __future__ import annotations

from aire.observability import JsonlExporter, MemoryExporter, Metrics, Tracer
from tests.conftest import arun


def test_tracer_nests_spans() -> None:
    exporter = MemoryExporter()
    tracer = Tracer(exporter=exporter)
    with tracer.span("parent"), tracer.span("child"):
        pass
    records = exporter.records
    assert len(records) == 2
    child = next(r for r in records if r.name == "child")
    root = next(r for r in records if r.name == "parent")
    assert child.parent_span_id == root.span_id
    assert child.trace_id == root.trace_id
    assert root.duration_ms >= 0


def test_tracer_records_errors() -> None:
    exporter = MemoryExporter()
    tracer = Tracer(exporter=exporter)
    try:
        with tracer.span("failing"):
            raise ValueError("boom")
    except ValueError:
        pass
    record = exporter.records[0]
    assert record.status == "error"
    assert "boom" in (record.error or "")


def test_tracer_masks_sensitive_fields() -> None:
    exporter = MemoryExporter()
    tracer = Tracer(exporter=exporter, mask_fields=["api_key"])
    with tracer.span("call", attributes={"api_key": "secret", "model": "gpt"}):
        pass
    attrs = exporter.records[0].attributes
    assert attrs["api_key"] == "***"
    assert attrs["model"] == "gpt"


def test_async_span() -> None:
    exporter = MemoryExporter()
    tracer = Tracer(exporter=exporter)

    async def _work() -> None:
        async with tracer.aspan("async-op"):
            pass

    arun(_work())
    assert exporter.records[0].name == "async-op"


def test_jsonl_exporter(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "traces.jsonl"
    tracer = Tracer(exporter=JsonlExporter(str(path)))
    with tracer.span("persisted"):
        pass
    import json

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["name"] == "persisted"


def test_metrics_snapshot() -> None:
    metrics = Metrics()
    metrics.increment("aire.tokens.input", 10, model="m")
    metrics.record_tokens(5, 7, model="m")
    metrics.record_cost(0.001, model="m")
    metrics.observe_latency("generate", 100.0)
    metrics.observe_latency("generate", 200.0)
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["aire.tokens.input{model=m}"] == 15.0
    assert snapshot["latencies"]["generate"]["count"] == 2
    assert snapshot["latencies"]["generate"]["max"] == 200.0
