import math

import pytest

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.evaluation.gates import check_gates
from aire.evaluation.types import CaseResult, EvalCase, EvalReport, MetricResult


def test_gates_accept_real_evaluation_report():
    report = EvalReport(
        results=[
            CaseResult(
                case=EvalCase(input="a"),
                output="a",
                metrics=[MetricResult(name="exact_match", score=1)],
            )
        ]
    )
    assert check_gates(report, [{"metric": "exact_match", "min": 1}]).passed
    assert not check_gates(EvalReport(), [{"metric": "exact_match", "min": 1}]).passed
    assert EvalReport().pass_rate() == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_metric_cannot_pass_release_gate(value):
    assert not check_gates({"score": value}, [{"metric": "score", "min": 0}]).passed


def test_dataset_version_changes_with_labels_but_dedup_remains_text_based():
    a = Dataset.from_dicts([{"text": "same", "label": "yes"}])
    b = Dataset.from_dicts([{"text": "same", "label": "no"}])
    assert a.version != b.version
    assert Dataset.from_texts(["same", "same"]).deduplicate().texts == ["same"]


def test_synthetic_count_is_explicit():
    from aire.models.base import run_sync
    from aire.models.builtin import EchoModel
    from aire.synthetic.generator import SyntheticGenerator

    generator = SyntheticGenerator(EchoModel())
    with pytest.raises(ConfigurationError, match="positive"):
        run_sync(generator.qa_pairs("content", n=0))
    with pytest.raises(ConfigurationError, match="requested"):
        run_sync(generator.qa_pairs("content", n=20))


def test_gates_can_reject_failed_cases_and_empty_evaluations():
    report = EvalReport(
        results=[CaseResult(case=EvalCase(input="a"), output="", error="provider failed")]
    )
    result = check_gates(report, [{"metric": "failures", "max": 0}])
    assert not result.passed and result.results[0].value == 1
    assert not check_gates(EvalReport(), [{"metric": "pass_rate", "min": 1}]).passed


def test_nonfinite_gate_threshold_is_rejected():
    from pydantic import ValidationError

    from aire.evaluation.gates import EvalGate

    with pytest.raises(ValidationError):
        EvalGate(metric="score", min=math.nan)
