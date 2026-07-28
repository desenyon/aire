"""Optional OpenTelemetry SDK bridge (deepens the HTTP OTLP exporter).

When the ``opentelemetry`` SDK is installed, spans can be mirrored into the
global TracerProvider. Without it, :class:`SdkBridgeExporter` is a no-op
wrapper around the existing :class:`~aire.observability.otlp.OTLPExporter`.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.observability.otlp import OTLPExporter
from aire.observability.tracing import SpanRecord


def otel_sdk_available() -> bool:
    return importlib.util.find_spec("opentelemetry") is not None


class SdkBridgeExporter:
    """Export aire :class:`SpanRecord`s to both OTLP/HTTP and the OTel SDK."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        service_name: str = "aire",
        otlp: OTLPExporter | None = None,
        use_sdk: bool = True,
        **otlp_options: Any,
    ) -> None:
        self.service_name = service_name
        self.otlp = otlp
        if endpoint is not None and self.otlp is None:
            self.otlp = OTLPExporter(endpoint, service_name=service_name, **otlp_options)
        self._tracer: Any = None
        self.sdk_exported = 0
        self.sdk_skipped = 0
        if use_sdk and otel_sdk_available():
            self._tracer = self._make_sdk_tracer()

    def _make_sdk_tracer(self) -> Any:
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "opentelemetry-api required for SDK bridge: "
                "pip install opentelemetry-api opentelemetry-sdk",
                code="observe.otel_sdk_missing",
                cause=exc,
            ) from exc
        return trace.get_tracer("aire", self.service_name)

    def export(self, record: SpanRecord) -> None:
        if self.otlp is not None:
            self.otlp.export(record)
        if self._tracer is None:
            self.sdk_skipped += 1
            return
        try:
            with self._tracer.start_as_current_span(record.name) as span:
                for key, value in record.attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
                    else:
                        span.set_attribute(key, str(value))
                if record.status == "error":
                    from opentelemetry.trace import (  # type: ignore[import-not-found]
                        Status,
                        StatusCode,
                    )

                    span.set_status(Status(StatusCode.ERROR, record.error or "error"))
            self.sdk_exported += 1
        except Exception:
            self.sdk_skipped += 1

    def flush(self) -> None:
        if self.otlp is not None:
            self.otlp.flush()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "otel_sdk_bridge",
            "sdk_available": otel_sdk_available(),
            "sdk_exported": self.sdk_exported,
            "sdk_skipped": self.sdk_skipped,
            "otlp": self.otlp.describe() if self.otlp else None,
            "install": "pip install opentelemetry-api opentelemetry-sdk",
        }


def create_exporter(
    endpoint: str | None = None,
    *,
    prefer_sdk: bool = True,
    **options: Any,
) -> OTLPExporter | SdkBridgeExporter:
    """Factory: SDK bridge when available, else plain OTLPExporter."""
    if prefer_sdk:
        return SdkBridgeExporter(endpoint, **options)
    if endpoint is None:
        raise ConfigurationError(
            "endpoint required for OTLPExporter",
            code="observe.otlp_endpoint_missing",
        )
    return OTLPExporter(endpoint, **options)
