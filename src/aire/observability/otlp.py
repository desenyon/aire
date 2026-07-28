"""OTLP/HTTP exporter: ship aire spans to any OpenTelemetry collector.

Speaks OTLP/HTTP + JSON (the collector's ``/v1/traces`` endpoint) over httpx —
no opentelemetry SDK dependency. Batched with a size trigger; export failures
are counted, never raised, so telemetry can never crash the application.

    tracer = Tracer(exporter=OTLPExporter("http://localhost:4318"))
    ...                                  # spans batch automatically
    tracer.exporter.flush()              # drain remaining spans
"""

from __future__ import annotations

from typing import Any

from aire.observability.tracing import SpanRecord

_NANOS = 1_000_000_000


def _to_otlp_span(record: SpanRecord) -> dict[str, Any]:
    span: dict[str, Any] = {
        "traceId": record.trace_id,
        "spanId": record.span_id,
        "name": record.name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": int(record.start_time * _NANOS),
        "endTimeUnixNano": int(record.end_time * _NANOS),
        "attributes": [
            {"key": key, "value": _otlp_value(value)} for key, value in record.attributes.items()
        ],
        "status": {"code": 2 if record.status == "error" else 1},
    }
    if record.parent_span_id:
        span["parentSpanId"] = record.parent_span_id
    if record.error:
        span["events"] = [
            {
                "timeUnixNano": int(record.end_time * _NANOS),
                "name": "exception",
                "attributes": [
                    {"key": "exception.message", "value": {"stringValue": record.error}}
                ],
            }
        ]
    return span


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_value(v) for v in value]}}
    return {"stringValue": str(value)}


class OTLPExporter:
    """Batched OTLP/HTTP+JSON span exporter."""

    def __init__(
        self,
        endpoint: str,
        *,
        service_name: str = "aire",
        headers: dict[str, str] | None = None,
        batch_size: int = 32,
        timeout: float = 5.0,
        client: Any = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.service_name = service_name
        self.headers = dict(headers or {})
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self._client = client  # injectable httpx.Client (tests use MockTransport)
        self._batch: list[SpanRecord] = []
        self.exported = 0
        self.failures = 0
        self.last_error: str | None = None

    def export(self, record: SpanRecord) -> None:
        self._batch.append(record)
        if len(self._batch) >= self.batch_size:
            self.flush()

    def payload(self, records: list[SpanRecord]) -> dict[str, Any]:
        """The OTLP/JSON request body for a batch (exposed for inspection)."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self.service_name},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "aire", "version": _aire_version()},
                            "spans": [_to_otlp_span(r) for r in records],
                        }
                    ],
                }
            ]
        }

    def flush(self) -> None:
        """POST the pending batch; failures are counted, never raised."""
        if not self._batch:
            return
        records, self._batch = self._batch, []
        try:
            client = self._client or self._make_client()
            response = client.post(
                f"{self.endpoint}/v1/traces",
                json=self.payload(records),
                headers={"content-type": "application/json", **self.headers},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"collector returned {response.status_code}")
            self.exported += len(records)
            self.last_error = None
        except Exception as exc:  # telemetry must never crash the app
            self.failures += len(records)
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _make_client(self) -> Any:
        import httpx

        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "otlp_exporter",
            "endpoint": self.endpoint,
            "service": self.service_name,
            "exported": self.exported,
            "failures": self.failures,
            "pending": len(self._batch),
        }


def _aire_version() -> str:
    from aire._version import __version__

    return __version__
