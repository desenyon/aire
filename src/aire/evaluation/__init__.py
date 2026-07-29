"""Evaluation system: datasets, metrics, judges, reports."""

from aire.evaluation.gates import EvalGate, GateReport, GateResult, check_gates
from aire.evaluation.metrics import get_metric, metric_names, register_metric
from aire.evaluation.runner import Evaluator, load_cases, save_report
from aire.evaluation.types import CaseResult, EvalCase, EvalReport, MetricResult

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalGate",
    "EvalReport",
    "Evaluator",
    "GateReport",
    "GateResult",
    "MetricResult",
    "check_gates",
    "get_metric",
    "load_cases",
    "metric_names",
    "register_metric",
    "save_report",
]
