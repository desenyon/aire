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
