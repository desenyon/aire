import httpx

from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.models.base import run_sync
from aire.observability.otlp import OTLPExporter
from aire.observability.tracing import Tracer


def test_otlp_hierarchy_encoding_and_runtime_flush():
    requests = []
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(200, json={}))
    )
    exporter = OTLPExporter("https://collector.example", client=client)
    tracer = Tracer(exporter=exporter)
    with tracer.span("parent"), tracer.span("child"):
        pass
    runtime = Runtime(Settings())
    runtime.tracer = tracer
    run_sync(runtime.aclose())
    import json

    payload = json.loads(requests[0].content)
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans[0]["parentSpanId"] == spans[1]["spanId"]
    assert spans[0]["traceId"] == spans[1]["traceId"]
    assert isinstance(spans[0]["startTimeUnixNano"], str)
    assert exporter.exported == 2


def test_export_failure_does_not_mask_application_error():
    import pytest

    exporter = OTLPExporter(
        "https://collector.example",
        batch_size=1,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    with pytest.raises(ValueError, match="application"), Tracer(exporter).span("fails"):
        raise ValueError("application")
    assert exporter.failures == 1


def test_otlp_partial_success_counts_rejections_without_retry():
    from aire.observability.tracing import SpanRecord

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"partialSuccess": {"rejectedSpans": "1", "errorMessage": "unsupported"}}
            )
        )
    )
    exporter = OTLPExporter("https://collector.example", client=client)
    for name in ["accepted", "rejected"]:
        exporter.export(
            SpanRecord(trace_id="a" * 32, span_id="b" * 16, name=name, start_time=1, end_time=2)
        )
    exporter.flush()
    assert exporter.exported == 1
    assert exporter.failures == 1
    assert not client.is_closed
