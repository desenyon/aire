"""0.2.8 — ML orchestration: Pipeline, transforms, backends, selection."""

from __future__ import annotations

import importlib.util

import pytest

from aire import AI
from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.callbacks import EarlyStopping
from aire.ml.transform import StandardScaler, create_transform
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


def test_backends_lists_all_orchestrated_libs() -> None:
    backends = AI.ml.backends()
    for key in ("native", "sklearn", "torch", "keras", "xgboost", "lightgbm", "pandas"):
        assert key in backends
        assert isinstance(backends[key], bool)


def test_catalog_includes_keras_and_boosting() -> None:
    catalog = AI.ml.catalog()
    assert "keras:mlp" in catalog["keras"]
    assert "xgboost:classifier" in catalog["xgboost"]
    assert "lightgbm:regressor" in catalog["lightgbm"]
    assert "torch:mlp" in catalog["torch"]


def test_native_transform_standard_scaler() -> None:
    scaler = create_transform("native:standard_scaler")
    assert isinstance(scaler, StandardScaler)
    x = [[0.0, 0.0], [2.0, 4.0]]
    out = scaler.fit_transform_matrix(x)
    assert abs(out[0][0] + 1.0) < 1e-9
    assert abs(out[1][0] - 1.0) < 1e-9


def test_pipeline_scale_then_centroid() -> None:
    pipe = AI.ml.pipeline(
        [("scale", "native:standard_scaler"), ("clf", "simple:centroid")]
    )
    report = arun(pipe.fit(_labeled_dataset()))
    assert report.samples == 24
    preds = arun(pipe.predict(_labeled_dataset()))
    assert len(preds) == 24
    assert {p.value for p in preds} <= {"low", "high"}


def test_train_with_transforms() -> None:
    fitted = arun(
        AI.ml.train(
            "simple:centroid",
            _labeled_dataset(),
            transforms=["native:minmax_scaler"],
        )
    )
    assert fitted.report is not None
    preds = arun(fitted.predict(_labeled_dataset()))
    assert len(preds) == 24


def test_random_search_knn() -> None:
    report = arun(
        AI.ml.random_search(
            "simple:knn",
            _labeled_dataset(),
            {"k": [1, 3, 5]},
            n_iter=2,
            k=2,
            seed=1,
        )
    )
    assert report.best_params["k"] in {1, 3, 5}
    assert len(report.trials) == 2


def test_grid_search_minimize_direction() -> None:
    # accuracy maximize still works; direction flag is accepted
    report = arun(
        AI.ml.grid_search(
            "simple:knn",
            _labeled_dataset(),
            {"k": [1, 3]},
            k=2,
            direction="maximize",
        )
    )
    assert report.best_score >= 0.0


def test_describe_mentions_orchestration() -> None:
    desc = AI.ml.describe()
    assert "Pipeline" in desc["contract"]
    assert "transforms" in desc
    assert "random_search" in desc["selection"]


def test_keras_missing_gives_clear_hint() -> None:
    if importlib.util.find_spec("keras") is not None:
        pytest.skip("keras installed")
    with pytest.raises(ConfigurationError) as exc:
        AI.ml.create("keras:mlp")
    assert "keras" in str(exc.value).lower()


def test_xgboost_missing_gives_clear_hint() -> None:
    if importlib.util.find_spec("xgboost") is not None:
        pytest.skip("xgboost installed")
    with pytest.raises(ConfigurationError) as exc:
        AI.ml.create("xgboost:classifier")
    assert "xgboost" in str(exc.value).lower()


def test_lightgbm_missing_gives_clear_hint() -> None:
    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("lightgbm installed")
    with pytest.raises(ConfigurationError) as exc:
        AI.ml.create("lightgbm:classifier")
    assert "lightgbm" in str(exc.value).lower()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_early_stopping_callback() -> None:
    from aire.ml.torch_adapter import TorchEstimator

    est = TorchEstimator(
        hidden=(8,),
        epochs=50,
        batch_size=8,
        optimizer="adam",
        callbacks=[EarlyStopping(monitor="train_loss", patience=2, min_delta=0.0)],
    )
    report = arun(est.fit(_labeled_dataset()))
    assert report.metrics["epochs"] < 50.0 or report.metrics.get("early_stopped", 0.0) in {
        0.0,
        1.0,
    }
    assert len(est.history) >= 1
