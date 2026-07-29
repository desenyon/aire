"""Evaluation gates offline."""

from types import SimpleNamespace

from aire.evaluation.gates import EvalGate, check_gates
from aire.evaluation.metrics import semantic_overlap
from aire.evaluation.types import EvalCase
from aire.models.base import run_sync


def main() -> None:
    case = EvalCase(input="q", expected="aire is offline friendly")
    ctx = SimpleNamespace(
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        context=None,
        judge=None,
        embedder=None,
    )
    metric = run_sync(semantic_overlap(case, "aire is offline friendly and fast", ctx))
    report = {"metrics": {metric.name: metric.score}}
    gates = check_gates(report, [EvalGate(metric="semantic_overlap", min=0.3)])
    print("metric:", metric.name, metric.score, metric.detail)
    print("gates passed:", gates.passed)
    print(gates.describe())


if __name__ == "__main__":
    main()
