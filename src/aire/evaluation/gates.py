"""Evaluation gates — fail a report when metrics miss thresholds."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError


class EvalGate(BaseModel):
    """A single metric threshold check."""

    metric: str
    min: float | None = None
    max: float | None = None
    required: bool = True

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GateResult(BaseModel):
    metric: str
    value: float | None = None
    passed: bool
    reason: str = ""


class GateReport(BaseModel):
    passed: bool
    results: list[GateResult] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        failed = [r for r in self.results if not r.passed]
        raise ConfigurationError(
            "evaluation gates failed",
            code="eval.gates_failed",
            context={
                "failed": [r.model_dump(mode="json") for r in failed],
                "missing": list(self.missing),
            },
        )


def _metric_map(report: Any) -> dict[str, float]:
    if isinstance(report, dict):
        metrics = report.get("metrics") or report
        if isinstance(metrics, dict):
            out: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    out[str(key)] = float(value)
                elif isinstance(value, dict) and "score" in value:
                    out[str(key)] = float(value["score"])
                elif isinstance(value, dict) and "mean" in value:
                    out[str(key)] = float(value["mean"])
            return out
    if hasattr(report, "metrics"):
        return _metric_map({"metrics": report.metrics})
    if hasattr(report, "model_dump"):
        return _metric_map(report.model_dump(mode="json"))
    raise ConfigurationError(
        "unsupported eval report for gates",
        code="eval.gates_report",
        context={"type": type(report).__name__},
    )


def check_gates(report: Any, gates: list[EvalGate] | list[dict[str, Any]]) -> GateReport:
    """Check an EvalReport (or metrics dict) against threshold gates."""
    parsed = [
        g if isinstance(g, EvalGate) else EvalGate.model_validate(g) for g in gates
    ]
    metrics = _metric_map(report)
    results: list[GateResult] = []
    missing: list[str] = []
    for gate in parsed:
        if gate.metric not in metrics:
            missing.append(gate.metric)
            results.append(
                GateResult(
                    metric=gate.metric,
                    value=None,
                    passed=not gate.required,
                    reason="missing metric",
                )
            )
            continue
        value = metrics[gate.metric]
        ok = True
        reasons: list[str] = []
        if gate.min is not None and value < gate.min:
            ok = False
            reasons.append(f"{value} < min {gate.min}")
        if gate.max is not None and value > gate.max:
            ok = False
            reasons.append(f"{value} > max {gate.max}")
        results.append(
            GateResult(
                metric=gate.metric,
                value=value,
                passed=ok,
                reason="; ".join(reasons) if reasons else "ok",
            )
        )
    return GateReport(passed=all(r.passed for r in results), results=results, missing=missing)
