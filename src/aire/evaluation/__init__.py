"""Evaluation system: datasets, metrics, judges, reports."""

from aire.evaluation.metrics import get_metric, metric_names, register_metric
from aire.evaluation.runner import Evaluator, load_cases, save_report
from aire.evaluation.types import CaseResult, EvalCase, EvalReport, MetricResult

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "Evaluator",
    "MetricResult",
    "get_metric",
    "load_cases",
    "metric_names",
    "register_metric",
    "save_report",
]
