"""Evaluate real outputs and demonstrate both passing and rejecting gates."""

from aire.core.errors import ConfigurationError
from aire.evaluation.gates import check_gates
from aire.evaluation.runner import Evaluator
from aire.models.base import run_sync
from aire.models.builtin import EchoModel


def main() -> None:
    report = run_sync(
        Evaluator().run(
            EchoModel(),
            [{"input": "offline answer", "expected": "offline answer"}],
            metrics=["exact_match"],
        )
    )
    passing = check_gates(report, [{"metric": "exact_match", "min": 1.0}])
    assert passing.passed
    rejected = check_gates(report, [{"metric": "missing-quality-measure", "min": 0.9}])
    assert not rejected.passed
    try:
        rejected.raise_if_failed()
    except ConfigurationError as exc:
        assert exc.code == "eval.gates_failed"
    else:
        raise AssertionError("A missing required metric must block release")
    print("Evaluation passed; missing-metric release gate rejected as expected.")


if __name__ == "__main__":
    main()
