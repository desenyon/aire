"""Tests for aire.ml model-creation orchestration (0.2.1)."""

from __future__ import annotations

import pytest

from aire import AI
from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml import TaskType, extract_features, vectorize
from aire.ml.native import CentroidClassifier, LinearRegressor, MajorityClassifier, create_native
from tests.conftest import arun


def _labeled_dataset() -> Dataset:
    records = []
    for i in range(12):
        records.append(
            Record(
                text=f"sample {i}",
                metadata={"features": {"x": float(i), "y": 0.0}, "label": "low"},
            )
        )
    for i in range(12):
        records.append(
            Record(
                text=f"sample hi {i}",
                metadata={"features": {"x": float(100 + i), "y": 1.0}, "label": "high"},
            )
        )
    return Dataset(records, name="toy")


def _regression_dataset() -> Dataset:
    records = [
        Record(
            text=f"row {i}",
            metadata={"features": {"x": float(i)}, "y": 3.0 * i + 7.0},
        )
        for i in range(40)
    ]
    return Dataset(records, name="reg")


# -- features ----------------------------------------------------------------------


def test_feature_convention_explicit_dict() -> None:
    record = Record(text="hello", metadata={"features": {"a": 1.0, "b": 2}})
    assert extract_features(record) == {"a": 1.0, "b": 2.0}


def test_feature_convention_numeric_metadata() -> None:
    record = Record(text="hi", metadata={"length": 3, "ratio": 0.5, "source": "x.txt"})
    features = extract_features(record)
    assert features == {"length": 3.0, "ratio": 0.5}


def test_feature_convention_text_fallback() -> None:
    record = Record(text="the quick brown fox jumps", metadata={})
    features = extract_features(record)
    assert features["token_count"] == 5.0
    assert features["char_count"] == float(len("the quick brown fox jumps"))


def test_vectorize_aligns_features() -> None:
    names, matrix = vectorize([{"a": 1.0}, {"b": 2.0, "a": 3.0}])
    assert names == ["a", "b"]
    assert matrix == [[1.0, 0.0], [3.0, 2.0]]


# -- native estimators ---------------------------------------------------------------


def test_majority_classifier_learns_and_persists(tmp_path) -> None:
    estimator = MajorityClassifier()
    report = arun(estimator.fit(_labeled_dataset()))
    assert report.samples == 24
    assert 0.0 < report.metrics["train_accuracy"] <= 1.0

    fresh = MajorityClassifier().load(estimator.save(tmp_path / "model.json"))
    predictions = arun(fresh.predict(_labeled_dataset()))
    assert len(predictions) == 24
    assert predictions[0].value in {"low", "high"}
    assert predictions[0].probabilities  # probability estimates available


def test_centroid_classifier_separates_toy_data() -> None:
    estimator = CentroidClassifier()
    arun(estimator.fit(_labeled_dataset()))
    metrics = arun(estimator.evaluate(_labeled_dataset()))
    assert metrics["accuracy"] == 1.0  # well-separated clusters


def test_knn_via_facade_with_options() -> None:
    estimator = AI.ml.create("simple:knn", k=1)
    arun(estimator.fit(_labeled_dataset()))
    metrics = arun(estimator.evaluate(_labeled_dataset()))
    assert metrics["accuracy"] == 1.0
    assert estimator.describe().kind == "estimator"


def test_linear_regression_fits_line() -> None:
    estimator = LinearRegressor(epochs=800, learning_rate=0.1)
    report = arun(estimator.fit(_regression_dataset(), target="y"))
    assert estimator.task == TaskType.REGRESSION
    metrics = arun(estimator.evaluate(_regression_dataset(), target="y"))
    assert metrics["mae"] < 0.5
    assert report.metrics["train_mse"] < 0.5


def test_unknown_native_estimator_raises() -> None:
    with pytest.raises(ConfigurationError):
        create_native("nope")


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ConfigurationError):
        arun(MajorityClassifier().predict(_labeled_dataset()))


def test_missing_target_raises() -> None:
    dataset = Dataset([Record(text="x", metadata={"features": {"a": 1.0}})], name="t")
    with pytest.raises(ConfigurationError):
        arun(MajorityClassifier().fit(dataset))


# -- facade -------------------------------------------------------------------------


def test_facade_fit_one_call() -> None:
    estimator = AI.ml.fit_sync("simple:centroid", _labeled_dataset())
    metrics = arun(estimator.evaluate(_labeled_dataset()))
    assert metrics["accuracy"] == 1.0


def test_facade_backends_report() -> None:
    backends = AI.ml.backends()
    assert backends["native"] is True
    assert set(backends) >= {"native", "sklearn", "torch", "pandas", "keras", "xgboost", "lightgbm"}


def test_facade_describe() -> None:
    manifest = AI.ml.describe()
    assert manifest["kind"] == "ml"
    assert any("knn" in r for r in manifest["estimators"]["simple"])
    catalog = AI.ml.catalog()
    assert "simple:centroid" in catalog["simple"]


def test_sklearn_missing_dep_hint() -> None:
    import importlib.util

    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("sklearn installed")
    with pytest.raises(ConfigurationError, match="pip install"):
        AI.ml.create("sklearn:random_forest")


def test_torch_missing_dep_hint() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed")
    with pytest.raises(ConfigurationError, match="pip install"):
        AI.ml.create("torch:mlp")


def test_pandas_missing_dep_hint() -> None:
    import importlib.util

    if importlib.util.find_spec("pandas") is not None:
        pytest.skip("pandas installed")
    with pytest.raises(ConfigurationError, match="pip install"):
        AI.ml.to_frame(_labeled_dataset())
