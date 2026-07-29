"""Powerful metrics analytics: reports, trends, and Prometheus export."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from aire.observability.metrics import Metrics


class CostReport(BaseModel):
    total_usd: float = 0.0
    by_model: dict[str, float] = Field(default_factory=dict)
    by_operation: dict[str, float] = Field(default_factory=dict)
    requests: int = 0
    avg_cost_usd: float = 0.0


class LatencyReport(BaseModel):
    operations: dict[str, dict[str, float]] = Field(default_factory=dict)


class AnalyticsReport(BaseModel):
    generated_at: float = Field(default_factory=time.time)
    costs: CostReport = Field(default_factory=CostReport)
    latency: LatencyReport = Field(default_factory=LatencyReport)
    counters: dict[str, float] = Field(default_factory=dict)
    gauges: dict[str, float] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class Analytics:
    """Analyze an in-process :class:`Metrics` registry into operator reports."""

    def __init__(self, metrics: Metrics | None = None) -> None:
        self.metrics = metrics or Metrics()
        self._history: list[AnalyticsReport] = []

    def record_run(
        self,
        *,
        model: str,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        operation: str = "generate",
    ) -> None:
        self.metrics.record_cost(cost_usd, model=model)
        self.metrics.record_tokens(input_tokens, output_tokens, model=model)
        self.metrics.observe_latency(f"aire.{operation}.latency_ms", latency_ms, model=model)
        self.metrics.increment(f"aire.{operation}.requests", 1.0, model=model)

    def report(self) -> AnalyticsReport:
        snap = self.metrics.snapshot()
        costs = CostReport()
        requests = 0
        for key, value in snap.get("counters", {}).items():
            if str(key).startswith("aire.cost.usd"):
                costs.total_usd += float(value)
                model = _label(str(key), "model")
                if model:
                    costs.by_model[model] = costs.by_model.get(model, 0.0) + float(value)
            if ".requests" in str(key):
                requests += int(value)
                op = str(key).split("{", 1)[0].removeprefix("aire.").removesuffix(".requests")
                costs.by_operation[op] = costs.by_operation.get(op, 0.0) + float(value)
        costs.requests = requests
        costs.avg_cost_usd = costs.total_usd / requests if requests else 0.0
        latency = LatencyReport(operations=dict(snap.get("latencies", {})))
        report = AnalyticsReport(
            costs=costs,
            latency=latency,
            counters=dict(snap.get("counters", {})),
            gauges=dict(snap.get("gauges", {})),
            summary={
                "total_usd": round(costs.total_usd, 6),
                "requests": requests,
                "models": sorted(costs.by_model),
                "top_model_cost": max(costs.by_model.items(), key=lambda kv: kv[1])[0]
                if costs.by_model
                else None,
            },
        )
        self._history.append(report)
        return report

    def history(self, *, limit: int = 20) -> list[AnalyticsReport]:
        return self._history[-limit:]

    def prometheus(self) -> str:
        """Render a Prometheus text exposition of current counters/gauges/latencies."""
        snap = self.metrics.snapshot()
        lines: list[str] = [
            "# HELP aire_analytics_info aire analytics scrape",
            "# TYPE aire_analytics_info gauge",
            'aire_analytics_info{library="aire"} 1',
        ]
        for key, value in sorted(snap.get("counters", {}).items()):
            metric, labels = _prom_name(str(key), "counter")
            lines.append(f"# TYPE {metric} counter")
            if labels:
                lines.append(f"{metric}{{{labels}}} {float(value)}")
            else:
                lines.append(f"{metric} {float(value)}")
        for key, value in sorted(snap.get("gauges", {}).items()):
            metric, labels = _prom_name(str(key), "gauge")
            lines.append(f"# TYPE {metric} gauge")
            if labels:
                lines.append(f"{metric}{{{labels}}} {float(value)}")
            else:
                lines.append(f"{metric} {float(value)}")
        for key, stats in sorted(snap.get("latencies", {}).items()):
            metric, labels = _prom_name(str(key), "latency")
            for stat_name, stat_val in stats.items():
                lab = f'{labels},stat="{stat_name}"' if labels else f'stat="{stat_name}"'
                lines.append(f"{metric}{{{lab}}} {float(stat_val)}")
        return "\n".join(lines) + "\n"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "analytics",
            "history": len(self._history),
            "methods": ["record_run", "report", "history", "prometheus"],
        }


def _label(key: str, name: str) -> str | None:
    marker = f"{name}="
    if marker not in key:
        return None
    return key.split(marker, 1)[1].rstrip("}")


def _prom_name(key: str, kind: str) -> tuple[str, str]:
    if "{" in key:
        base, rest = key.split("{", 1)
        labels = rest.rstrip("}")
    else:
        base, labels = key, ""
    metric = base.replace(".", "_").replace("-", "_")
    if not metric.startswith("aire_"):
        metric = f"aire_{kind}_{metric}"
    return metric, labels


def create_analytics(metrics: Metrics | None = None) -> Analytics:
    return Analytics(metrics)
