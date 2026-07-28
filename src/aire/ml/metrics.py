"""ML evaluation helpers: classification reports, CV, grid search, importances."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.types import TaskType


class ClassificationReport(BaseModel):
    """Per-class precision / recall / F1 plus micro/macro averages."""

    accuracy: float
    samples: int
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0

    def as_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "accuracy": self.accuracy,
            "samples": float(self.samples),
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
        }
        for label, scores in self.per_class.items():
            for key, value in scores.items():
                out[f"{label}.{key}"] = value
        return out


class CVFold(BaseModel):
    fold: int
    train_samples: int
    test_samples: int
    metrics: dict[str, float] = Field(default_factory=dict)


class CVReport(BaseModel):
    folds: list[CVFold]
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    duration_s: float = 0.0


class GridSearchReport(BaseModel):
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    scoring: str = "accuracy"
    trials: list[dict[str, Any]] = Field(default_factory=list)
    duration_s: float = 0.0


def classification_report(truth: list[Any], predictions: list[Any]) -> ClassificationReport:
    """Compute a full classification report from parallel truth/pred sequences."""
    if len(truth) != len(predictions):
        raise ConfigurationError(
            "truth and predictions length mismatch",
            code="ml.length_mismatch",
            context={"truth": len(truth), "predictions": len(predictions)},
        )
    y_true = [str(t) for t in truth]
    y_pred = [str(p) for p in predictions]
    labels = sorted(set(y_true) | set(y_pred))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)
    correct = 0
    for t, p in zip(y_true, y_pred, strict=True):
        support[t] += 1
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(support[label]),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    n = len(y_true)
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return ClassificationReport(
        accuracy=correct / n if n else 0.0,
        samples=n,
        per_class=per_class,
        macro_precision=sum(precisions) / len(precisions) if precisions else 0.0,
        macro_recall=sum(recalls) / len(recalls) if recalls else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
    )


def regression_metrics(truth: list[Any], predictions: list[Any]) -> dict[str, float]:
    """MAE, RMSE, R² for regression predictions."""
    if not truth:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "samples": 0.0}
    errors = [float(p) - float(t) for p, t in zip(predictions, truth, strict=True)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
    mean_t = sum(float(t) for t in truth) / len(truth)
    ss_tot = sum((float(t) - mean_t) ** 2 for t in truth)
    ss_res = sum(e * e for e in errors)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "samples": float(len(truth))}


def _kfold_indices(n: int, k: int, *, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    if k < 2:
        raise ConfigurationError("k must be >= 2", code="ml.cv_k")
    if n < k:
        raise ConfigurationError(
            f"need at least {k} samples for {k}-fold CV, got {n}",
            code="ml.cv_samples",
        )
    order = list(range(n))
    # deterministic shuffle
    rng = _LCG(seed)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    folds: list[tuple[list[int], list[int]]] = []
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    start = 0
    for size in fold_sizes:
        test = order[start : start + size]
        train = order[:start] + order[start + size :]
        folds.append((train, test))
        start += size
    return folds


class _LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def randint(self, lo: int, hi: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0xFFFFFFFF
        return lo + (self.state % (hi - lo + 1))


def _subset(records: list[Record], indices: list[int], name: str) -> Dataset:
    return Dataset(name=name, records=[records[i] for i in indices])


def confusion_matrix(truth: list[Any], predictions: list[Any]) -> dict[str, Any]:
    """Return labels + matrix counts (rows=true, cols=pred)."""
    y_true = [str(t) for t in truth]
    y_pred = [str(p) for p in predictions]
    labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return {"labels": labels, "matrix": matrix}


def _stratified_kfold_indices(  # noqa: C901
    labels: list[Any], k: int, *, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Stratified k-fold: preserve class proportions in each fold."""
    if k < 2:
        raise ConfigurationError("k must be >= 2", code="ml.cv_k")
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[str(label)].append(i)
    rng = _LCG(seed)
    for idxs in by_class.values():
        for i in range(len(idxs) - 1, 0, -1):
            j = rng.randint(0, i)
            idxs[i], idxs[j] = idxs[j], idxs[i]
    fold_train: list[list[int]] = [[] for _ in range(k)]
    fold_test: list[list[int]] = [[] for _ in range(k)]
    for idxs in by_class.values():
        if len(idxs) < k:
            # fall back: put all in every train, cycle test
            for i, idx in enumerate(idxs):
                fold_test[i % k].append(idx)
                for f in range(k):
                    if f != i % k:
                        fold_train[f].append(idx)
            continue
        sizes = [len(idxs) // k + (1 if i < len(idxs) % k else 0) for i in range(k)]
        start = 0
        for f, size in enumerate(sizes):
            test = idxs[start : start + size]
            train = idxs[:start] + idxs[start + size :]
            fold_test[f].extend(test)
            fold_train[f].extend(train)
            start += size
    return list(zip(fold_train, fold_test, strict=True))


async def cross_validate(
    factory: Any,
    dataset: Dataset,
    *,
    k: int = 5,
    target: str = "label",
    seed: int = 0,
    scoring: str | None = None,
    stratified: bool = False,
) -> CVReport:
    """K-fold cross-validation; ``factory()`` must return a fresh Estimator."""
    started = time.time()
    records = list(dataset)
    if stratified:
        labels = [record.metadata.get(target) for record in records]
        folds_idx = _stratified_kfold_indices(labels, k, seed=seed)
    else:
        folds_idx = _kfold_indices(len(records), k, seed=seed)
    fold_reports: list[CVFold] = []
    for i, (train_i, test_i) in enumerate(folds_idx):
        est = factory()
        train_ds = _subset(records, train_i, f"cv-train-{i}")
        test_ds = _subset(records, test_i, f"cv-test-{i}")
        await est.fit(train_ds, target=target)
        metrics = await est.evaluate(test_ds, target=target)
        if scoring:
            from aire.ml.scoring import score as _score

            preds = await est.predict(list(test_ds))
            truth = [r.metadata.get(target) for r in test_ds]
            probs = [p.probabilities or {} for p in preds]
            metrics = {
                **metrics,
                scoring: _score(scoring, truth, [p.value for p in preds], probs),
            }
        fold_reports.append(
            CVFold(
                fold=i,
                train_samples=len(train_i),
                test_samples=len(test_i),
                metrics=dict(metrics),
            )
        )
    keys = sorted({key for fold in fold_reports for key in fold.metrics})
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = [fold.metrics.get(key, 0.0) for fold in fold_reports]
        m = sum(values) / len(values)
        mean[key] = m
        var = sum((v - m) ** 2 for v in values) / len(values)
        std[key] = math.sqrt(var)
    return CVReport(folds=fold_reports, mean=mean, std=std, duration_s=time.time() - started)


async def grid_search(
    factory: Any,
    dataset: Dataset,
    param_grid: dict[str, list[Any]],
    *,
    k: int = 3,
    target: str = "label",
    scoring: str | None = None,
    direction: str = "maximize",
    seed: int = 0,
) -> GridSearchReport:
    """Exhaustive grid search with inner k-fold CV; returns best params + trials."""
    started = time.time()
    keys = sorted(param_grid)
    if not keys:
        raise ConfigurationError("param_grid is empty", code="ml.empty_grid")
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError(
            f"direction must be maximize|minimize, got {direction!r}",
            code="ml.direction",
        )
    trials: list[dict[str, Any]] = []
    best_score = float("-inf") if direction == "maximize" else float("inf")
    best_params: dict[str, Any] = {}

    def _product(i: int, current: dict[str, Any]) -> None:
        if i == len(keys):
            trials.append(dict(current))
            return
        key = keys[i]
        for value in param_grid[key]:
            current[key] = value
            _product(i + 1, current)

    _product(0, {})

    scored: list[dict[str, Any]] = []
    for params in trials:
        captured = dict(params)

        def make(captured: dict[str, Any] = captured) -> Any:
            return factory(**captured)

        report = await cross_validate(make, dataset, k=k, target=target, seed=seed)
        metric_key = scoring or _default_score_key(report.mean)
        score = report.mean.get(metric_key, float("-inf"))
        entry = {
            "params": params,
            "score": score,
            "scoring": metric_key,
            "cv_mean": report.mean,
        }
        scored.append(entry)
        better = score > best_score if direction == "maximize" else score < best_score
        if better:
            best_score = score
            best_params = dict(params)

    return GridSearchReport(
        best_params=best_params,
        best_score=best_score,
        scoring=scoring or (scored[0]["scoring"] if scored else "accuracy"),
        trials=scored,
        duration_s=time.time() - started,
    )


async def random_search(
    factory: Any,
    dataset: Dataset,
    param_distributions: dict[str, list[Any]],
    *,
    n_iter: int = 10,
    k: int = 3,
    target: str = "label",
    scoring: str | None = None,
    direction: str = "maximize",
    seed: int = 0,
) -> GridSearchReport:
    """Sample ``n_iter`` parameter combinations from discrete distributions."""
    started = time.time()
    keys = sorted(param_distributions)
    if not keys:
        raise ConfigurationError("param_distributions is empty", code="ml.empty_grid")
    rng = _LCG(seed)
    trials: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = n_iter * 20
    while len(trials) < n_iter and attempts < max_attempts:
        attempts += 1
        params = {
            key: param_distributions[key][
                rng.randint(0, len(param_distributions[key]) - 1)
            ]
            for key in keys
        }
        key_t = tuple((k, params[k]) for k in keys)
        if key_t in seen:
            continue
        seen.add(key_t)
        trials.append(params)

    best_score = float("-inf") if direction == "maximize" else float("inf")
    best_params: dict[str, Any] = {}
    scored: list[dict[str, Any]] = []
    for params in trials:
        captured = dict(params)

        def make(captured: dict[str, Any] = captured) -> Any:
            return factory(**captured)

        report = await cross_validate(make, dataset, k=k, target=target, seed=seed)
        metric_key = scoring or _default_score_key(report.mean)
        score = report.mean.get(metric_key, float("-inf"))
        scored.append(
            {"params": params, "score": score, "scoring": metric_key, "cv_mean": report.mean}
        )
        better = score > best_score if direction == "maximize" else score < best_score
        if better:
            best_score = score
            best_params = dict(params)

    return GridSearchReport(
        best_params=best_params,
        best_score=best_score,
        scoring=scoring or (scored[0]["scoring"] if scored else "accuracy"),
        trials=scored,
        duration_s=time.time() - started,
    )


def _default_score_key(metrics: dict[str, float]) -> str:
    if "accuracy" in metrics:
        return "accuracy"
    if "macro_f1" in metrics:
        return "macro_f1"
    if "r2" in metrics:
        return "r2"
    if "rmse" in metrics:
        return "rmse"
    return next(iter(metrics), "accuracy")


async def permutation_importance(
    estimator: Any,
    dataset: Dataset,
    *,
    target: str = "label",
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation feature importance: drop in score when a feature is shuffled."""
    records = list(dataset)
    if not records or not estimator.feature_names:
        return {}
    baseline = await estimator.evaluate(dataset, target=target)
    score_key = "accuracy" if estimator.task == TaskType.CLASSIFICATION else "r2"
    # for rmse, lower is better — convert to negative for drop computation
    if score_key not in baseline and "rmse" in baseline:
        baseline_score = -baseline["rmse"]
        use_neg_rmse = True
    else:
        baseline_score = baseline.get(score_key, 0.0)
        use_neg_rmse = False
    importances: dict[str, float] = {}
    rng = _LCG(seed)
    for feat_i, name in enumerate(estimator.feature_names):
        drops: list[float] = []
        for _ in range(n_repeats):
            shuffled = _shuffle_feature(records, name, rng)
            metrics = await estimator.evaluate(
                Dataset(name="perm", records=shuffled), target=target
            )
            score = -metrics["rmse"] if use_neg_rmse else metrics.get(score_key, 0.0)
            drops.append(baseline_score - score)
        importances[name] = sum(drops) / len(drops)
        _ = feat_i
    return importances


def _shuffle_feature(records: list[Record], feature: str, rng: _LCG) -> list[Record]:
    values: list[float] = []
    for record in records:
        feats = record.metadata.get("features")
        if isinstance(feats, dict) and feature in feats:
            values.append(float(feats[feature]))
        elif feature in record.metadata and isinstance(record.metadata[feature], (int, float)):
            values.append(float(record.metadata[feature]))
        else:
            values.append(0.0)
    order = list(range(len(values)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    shuffled_vals = [values[i] for i in order]
    out: list[Record] = []
    for record, new_val in zip(records, shuffled_vals, strict=True):
        meta = dict(record.metadata)
        feats = meta.get("features")
        if isinstance(feats, dict):
            feats = dict(feats)
            feats[feature] = new_val
            meta["features"] = feats
        else:
            meta[feature] = new_val
        out.append(Record(id=record.id, text=record.text, metadata=meta))
    return out
