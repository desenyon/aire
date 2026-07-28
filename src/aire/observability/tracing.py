"""Tracing with OpenTelemetry-compatible field names.

Spans propagate through :mod:`contextvars`, so nested operations (agent → tool
→ model call) automatically form a tree without explicit passing. Exporters
are pluggable: in-memory (default, for tests/UI), JSONL file, or a real OTLP
exporter via the plugin system.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

_current_span: ContextVar[Span | None] = ContextVar("aire_span", default=None)
_current_trace: ContextVar[str | None] = ContextVar("aire_trace", default=None)


class SpanRecord(BaseModel):
    """A finished span, serialized for export."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    start_time: float
    end_time: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0


class Span:
    """A live span. Always created through a :class:`Tracer`."""

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent: Span | None = None,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.parent = parent
        self.start_time = time.time()
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.status = "ok"
        self.error: str | None = None
        self._token_span: Any = None
        self._token_trace: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_error(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> Span:
        self._token_span = _current_span.set(self)
        self._token_trace = _current_trace.set(self.trace_id)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if exc is not None:
            self.record_error(exc)
        _current_span.reset(self._token_span)
        _current_trace.reset(self._token_trace)
        self.tracer._finish(self)

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.__exit__(exc_type, exc, tb)

    def finish(self) -> SpanRecord:
        return SpanRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent.span_id if self.parent else None,
            name=self.name,
            start_time=self.start_time,
            end_time=time.time(),
            attributes=dict(self.attributes),
            status=self.status,
            error=self.error,
        )


@runtime_checkable
class SpanExporter(Protocol):
    def export(self, record: SpanRecord) -> None: ...


class MemoryExporter:
    """Keeps finished spans in memory (default; inspect via ``AI.observe``)."""

    def __init__(self, limit: int = 10_000) -> None:
        self.records: list[SpanRecord] = []
        self.limit = limit

    def export(self, record: SpanRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.limit:
            del self.records[: len(self.records) - self.limit]

    def clear(self) -> None:
        self.records.clear()


class JsonlExporter:
    """Appends finished spans to a JSONL file."""

    def __init__(self, path: str) -> None:
        from pathlib import Path

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, record: SpanRecord) -> None:
        with self.path.open("a") as fh:
            fh.write(record.model_dump_json() + "\n")


class Tracer:
    """Creates spans and forwards finished ones to an exporter."""

    def __init__(
        self, exporter: SpanExporter | None = None, *, mask_fields: list[str] | None = None
    ) -> None:
        self.exporter = exporter or MemoryExporter()
        self.mask_fields = {f.lower() for f in mask_fields or []}

    @contextmanager
    def span(self, name: str, *, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            self,
            name,
            attributes=self._mask(attributes),
            trace_id=parent.trace_id if parent else _current_trace.get(),
            parent=parent,
        )
        with span:
            yield span

    @asynccontextmanager
    async def aspan(
        self, name: str, *, attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[Span]:
        with self.span(name, attributes=attributes) as span:
            yield span

    # alias used by subsystems: tracer.span works in both sync/async contexts
    def _finish(self, span: Span) -> None:
        self.exporter.export(span.finish())

    def _mask(self, attributes: dict[str, Any] | None) -> dict[str, Any] | None:
        if not attributes or not self.mask_fields:
            return attributes
        return {k: ("***" if k.lower() in self.mask_fields else v) for k, v in attributes.items()}

    def current_trace_id(self) -> str | None:
        return _current_trace.get()

    def records(self) -> list[SpanRecord]:
        if isinstance(self.exporter, MemoryExporter):
            return list(self.exporter.records)
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "tracer",
            "exporter": type(self.exporter).__name__,
            "spans": len(self.records()),
        }
