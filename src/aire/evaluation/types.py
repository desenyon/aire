"""Evaluation primitives: cases, metric results, reports."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import Usage, new_id


class EvalCase(BaseModel):
    """One evaluation example.

    Loaded from JSONL where each line has at least ``input`` and optionally
    ``expected``, ``context`` (for grounding metrics) and arbitrary metadata.
    """

    id: str = Field(default_factory=lambda: new_id("case"))
    input: str
    expected: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """One metric's score for one case."""

    name: str
    score: float  # 0.0 - 1.0 unless the metric documents otherwise
    detail: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """Everything preserved about one evaluated execution."""

    case: EvalCase
    output: str
    metrics: list[MetricResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None
    error_category: str | None = None
    model: str = "unknown"
    config: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    trace_id: str | None = None

    @property
    def mean_score(self) -> float:
        scores = [m.score for m in self.metrics]
        return sum(scores) / len(scores) if scores else 0.0


class EvalReport(BaseModel):
    """Aggregate outcome of an evaluation run."""

    id: str = Field(default_factory=lambda: new_id("eval"))
    name: str = "evaluation"
    target: str = "unknown"
    results: list[CaseResult] = Field(default_factory=list)
    started: float = Field(default_factory=time.time)
    finished: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if r.error)

    def metric_summary(self) -> dict[str, dict[str, float]]:
        """Mean/min/max per metric across cases."""
        buckets: dict[str, list[float]] = {}
        for result in self.results:
            for metric in result.metrics:
                buckets.setdefault(metric.name, []).append(metric.score)
        return {
            name: {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }
            for name, scores in sorted(buckets.items())
        }

    def pass_rate(self, threshold: float = 0.5) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if not r.error and r.mean_score >= threshold)
        return passed / len(self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "total": self.total,
            "failures": self.failures,
            "pass_rate": round(self.pass_rate(), 4),
            "metrics": self.metric_summary(),
            "duration_s": round(max(self.finished - self.started, 0.0), 3),
        }
