"""Evaluation metrics, runner and reports."""

from __future__ import annotations

from pathlib import Path

from aire.evaluation import EvalCase, Evaluator, load_cases, save_report
from aire.evaluation.metrics import get_metric, metric_names, register_metric
from aire.models.builtin import EchoModel
from tests.conftest import arun


def test_builtin_metrics_registered() -> None:
    names = metric_names()
    for expected in ("exact_match", "accuracy", "groundedness", "latency", "cost", "model_judge"):
        assert expected in names


def test_exact_match_metric() -> None:
    metric = get_metric("exact_match")
    case = EvalCase(input="q", expected="answer")

    class _Ctx:
        latency_ms = 1.0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        context = None
        judge = None

    assert arun(metric(case, "answer", _Ctx())).score == 1.0
    assert arun(metric(case, "wrong", _Ctx())).score == 0.0


def test_groundedness_metric() -> None:
    metric = get_metric("groundedness")
    case = EvalCase(input="q", context="refunds allowed within thirty days")

    class _Ctx:
        latency_ms = 1.0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        context = case.context
        judge = None

    grounded = arun(metric(case, "refunds are allowed within thirty days", _Ctx()))
    ungrounded = arun(metric(case, "quantum entanglement enables teleportation", _Ctx()))
    assert grounded.score > ungrounded.score


def test_custom_metric_registration() -> None:
    async def length_metric(case: EvalCase, output: str, ctx: object) -> object:
        from aire.evaluation.types import MetricResult

        return MetricResult(name="length", score=min(1.0, len(output) / 100))

    register_metric("length", length_metric)
    assert "length" in metric_names()


def test_evaluator_over_model(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"input": "one", "expected": "one"}\n{"input": "two", "expected": "different"}\n'
    )
    report = arun(Evaluator().run(EchoModel(), dataset, metrics=["exact_match"]))
    assert report.total == 2
    summary = report.metric_summary()
    assert summary["exact_match"]["mean"] == 0.5
    assert report.results[0].output == "one"


def test_evaluator_preserves_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"input": "ping", "expected": "ping"}\n')
    report = arun(Evaluator().run(EchoModel(), dataset, metrics=["exact_match", "latency"]))
    result = report.results[0]
    assert result.case.input == "ping"
    assert result.output == "ping"
    assert result.latency_ms >= 0
    assert result.usage.input_tokens >= 0
    assert result.timestamp > 0


def test_evaluator_records_errors(tmp_path: Path) -> None:
    def broken(input: str) -> str:
        raise RuntimeError("kaboom")

    report = arun(Evaluator().run(broken, [{"input": "x"}], metrics=["exact_match"]))
    assert report.failures == 1
    assert report.results[0].error_category == "aire.internal"


def test_evaluator_with_callable_target() -> None:
    report = arun(Evaluator().run(lambda q: q.upper(), [{"input": "a", "expected": "A"}]))
    assert report.results[0].output == "A"
    assert report.pass_rate() == 1.0


def test_load_cases_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"input": "q", "expected": "a", "metadata": {"k": 1}}\n')
    cases = load_cases(path)
    assert cases[0].metadata["k"] == 1


def test_save_report(tmp_path: Path) -> None:
    report = arun(Evaluator().run(EchoModel(), [{"input": "x", "expected": "x"}]))
    out = save_report(report, tmp_path / "report.json")
    import json

    data = json.loads(out.read_text())
    assert data["results"][0]["output"] == "x"
