"""Scoring registry — named metrics for CV / search / evaluate."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from aire.core.errors import ConfigurationError
from aire.ml.metrics import classification_report, regression_metrics

Scorer = Callable[[list[Any], list[Any], list[dict[str, float]] | None], float]


def _accuracy(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    if not y_true:
        return 0.0
    return sum(str(t) == str(p) for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def _macro_f1(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    return classification_report(y_true, y_pred).macro_f1


def _r2(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    return regression_metrics(y_true, y_pred)["r2"]


def _neg_rmse(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    return -regression_metrics(y_true, y_pred)["rmse"]


def _mae(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    return regression_metrics(y_true, y_pred)["mae"]


def _log_loss(
    y_true: list[Any], y_pred: list[Any], probabilities: list[dict[str, float]] | None
) -> float:
    if not probabilities:
        raise ConfigurationError(
            "log_loss requires predict_proba support", code="ml.score_proba"
        )
    eps = 1e-15
    total = 0.0
    for t, probs in zip(y_true, probabilities, strict=True):
        p = max(min(probs.get(str(t), eps), 1.0 - eps), eps)
        total += -math.log(p)
    return total / len(y_true)


def _roc_auc(
    y_true: list[Any], y_pred: list[Any], probabilities: list[dict[str, float]] | None
) -> float:
    """Binary ROC-AUC from positive-class probabilities (Mann-Whitney)."""
    _ = y_pred
    if not probabilities:
        raise ConfigurationError("roc_auc requires probabilities", code="ml.score_proba")
    labels = sorted({str(t) for t in y_true})
    if len(labels) != 2:
        raise ConfigurationError(
            "roc_auc scoring requires exactly 2 classes",
            code="ml.roc_binary",
            context={"classes": labels},
        )
    pos = labels[1]
    scores = [
        (1 if str(t) == pos else 0, float(probs.get(pos, 0.0)))
        for t, probs in zip(y_true, probabilities, strict=True)
    ]
    pos_scores = [s for y, s in scores if y == 1]
    neg_scores = [s for y, s in scores if y == 0]
    if not pos_scores or not neg_scores:
        return 0.0
    # Mann-Whitney U / (n_pos * n_neg)
    better = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                better += 1.0
            elif ps == ns:
                better += 0.5
    return better / (len(pos_scores) * len(neg_scores))


def _balanced_accuracy(y_true: list[Any], y_pred: list[Any], _: Any) -> float:
    report = classification_report(y_true, y_pred)
    recalls = [v["recall"] for v in report.per_class.values()]
    return sum(recalls) / len(recalls) if recalls else 0.0


_SCORERS: dict[str, tuple[Scorer, str]] = {
    # name -> (fn, direction maximize|minimize)
    "accuracy": (_accuracy, "maximize"),
    "macro_f1": (_macro_f1, "maximize"),
    "balanced_accuracy": (_balanced_accuracy, "maximize"),
    "r2": (_r2, "maximize"),
    "neg_rmse": (_neg_rmse, "maximize"),
    "mae": (_mae, "minimize"),
    "rmse": (lambda yt, yp, p: regression_metrics(yt, yp)["rmse"], "minimize"),
    "log_loss": (_log_loss, "minimize"),
    "roc_auc": (_roc_auc, "maximize"),
}


def register_scorer(
    name: str, fn: Scorer, *, direction: str = "maximize", replace: bool = False
) -> None:
    if name in _SCORERS and not replace:
        raise ConfigurationError(f"scorer {name!r} already registered", code="ml.scorer_dup")
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError("direction must be maximize|minimize", code="ml.direction")
    _SCORERS[name] = (fn, direction)


def score(
    name: str,
    y_true: list[Any],
    y_pred: list[Any],
    probabilities: list[dict[str, float]] | None = None,
) -> float:
    if name not in _SCORERS:
        raise ConfigurationError(
            f"unknown scorer {name!r}",
            code="ml.scorer_unknown",
            context={"available": sorted(_SCORERS)},
        )
    return _SCORERS[name][0](y_true, y_pred, probabilities)


def scorer_direction(name: str) -> str:
    if name not in _SCORERS:
        raise ConfigurationError(f"unknown scorer {name!r}", code="ml.scorer_unknown")
    return _SCORERS[name][1]


def scorers() -> dict[str, str]:
    """Map scorer name → preferred direction."""
    return {name: direction for name, (_, direction) in sorted(_SCORERS.items())}
