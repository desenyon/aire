"""0.2.9 — deep ML orchestration across sklearn / torch / keras ecosystems."""

from __future__ import annotations

import importlib.util

import pytest

from aire import AI
from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.compose import ColumnTransformer, FeatureUnion
from aire.ml.metrics import confusion_matrix
from aire.ml.scoring import score, scorers
from aire.ml.transform import StandardScaler
from tests.conftest import arun


def _labeled_dataset() -> Dataset:
    records = []
    for i in range(12):
        records.append(
            Record(
                text=f"sample {i}",
                metadata={"features": {"x": float(i), "y": 0.0, "z": 1.0}, "label": "low"},
            )
        )
    for i in range(12):
        records.append(
            Record(
                text=f"sample hi {i}",
                metadata={
                    "features": {"x": float(100 + i), "y": 1.0, "z": 2.0},
                    "label": "high",
                },
            )
        )
    return Dataset(records, name="toy")


def test_scorers_registry() -> None:
    names = scorers()
    assert "accuracy" in names
    assert "roc_auc" in names
    assert names["mae"] == "minimize"
    assert score("accuracy", ["a", "b"], ["a", "b"]) == 1.0


def test_confusion_matrix() -> None:
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert cm["labels"] == ["a", "b"]
    assert cm["matrix"][0][0] == 1


def test_column_transformer_and_feature_union() -> None:
    ct = ColumnTransformer(
        [
            ("sx", StandardScaler(), ["x"]),
            ("sy", "native:minmax_scaler", ["y"]),
        ],
        remainder="passthrough",
    )
    ct.feature_names = ["x", "y", "z"]
    x = [[0.0, 0.0, 5.0], [2.0, 1.0, 5.0]]
    out = ct.fit_transform_matrix(x)
    assert len(out[0]) == 3  # x scaled + y scaled + z passthrough

    fu = FeatureUnion(
        [("a", StandardScaler()), ("b", "native:minmax_scaler")]
    )
    fu.feature_names = ["x", "y"]
    out2 = fu.fit_transform_matrix([[0.0, 0.0], [2.0, 4.0]])
    assert len(out2[0]) == 4


def test_facade_column_transformer() -> None:
    ct = AI.ml.column_transformer(
        [("sx", "native:standard_scaler", ["x", "y"])],
        remainder="drop",
    )
    assert isinstance(ct, ColumnTransformer)


def test_catalog_includes_catboost_and_clustering() -> None:
    catalog = AI.ml.catalog()
    assert "catboost:classifier" in catalog["catboost"]
    assert "sklearn:kmeans" in catalog.get("sklearn", []) or not AI.ml.backends()["sklearn"]


def test_backends_include_catboost_polars() -> None:
    backends = AI.ml.backends()
    assert "catboost" in backends
    assert "polars" in backends


def test_stratified_cv() -> None:
    report = arun(
        AI.ml.cross_validate(
            "simple:centroid",
            _labeled_dataset(),
            k=3,
            stratified=True,
            seed=0,
        )
    )
    assert len(report.folds) == 3
    assert "accuracy" in report.mean


def test_scorers_on_describe() -> None:
    desc = AI.ml.describe()
    assert "roc_auc" in desc["scorers"]
    assert "clustering" in desc["tasks"]
    assert "AI.ml.column_transformer" in desc["orchestration"]


def test_catboost_missing_hint() -> None:
    if importlib.util.find_spec("catboost") is not None:
        pytest.skip("catboost installed")
    with pytest.raises(ConfigurationError) as exc:
        AI.ml.create("catboost:classifier")
    assert "catboost" in str(exc.value).lower()


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn missing")
def test_sklearn_partial_fit_and_importances() -> None:
    est = AI.ml.create("sklearn:sgd_classifier", max_iter=5)
    ds = _labeled_dataset()
    arun(est.fit(ds))
    # partial_fit on fresh estimator
    est2 = AI.ml.create("sklearn:sgd_classifier", max_iter=5)
    rows = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    from aire.ml.types import vectorize

    names, x = vectorize(rows)
    est2.feature_names = names
    est2.partial_fit(x, ["low", "high"], classes=["low", "high"])
    assert est2.report is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")
def test_torch_validation_and_proba() -> None:
    from aire.ml.callbacks import EarlyStopping

    est = AI.ml.create(
        "torch:mlp",
        hidden=(8,),
        epochs=5,
        batch_size=8,
        validation_split=0.25,
        callbacks=[EarlyStopping(monitor="train_loss", patience=10)],
    )
    report = arun(est.fit(_labeled_dataset()))
    assert "train_loss" in report.metrics
    preds = arun(est.predict(_labeled_dataset()))
    assert preds[0].probabilities
    assert sum(preds[0].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
