import importlib.util

import pytest

from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.factory import create_estimator
from aire.models.base import run_sync

pytestmark = pytest.mark.optional


@pytest.mark.parametrize(
    "backend,reference,options",
    [
        ("sklearn", "sklearn:logistic_regression", {}),
        ("xgboost", "xgboost:classifier", {"n_estimators": 2, "max_depth": 1, "n_jobs": 1}),
        ("lightgbm", "lightgbm:classifier", {"n_estimators": 2, "verbosity": -1, "n_jobs": 1}),
        ("catboost", "catboost:classifier", {"iterations": 2, "thread_count": 1, "verbose": False}),
        ("torch", "torch:mlp", {"epochs": 2, "hidden": (4,), "batch_size": 4}),
        ("keras", "keras:mlp", {"epochs": 2, "hidden": (4,), "batch_size": 4, "verbose": 0}),
    ],
)
def test_real_backend_fit_predict_save_load(backend, reference, options, tmp_path, monkeypatch):
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"install {backend} profile")
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    dataset = Dataset(
        [
            Record(text=str(i), metadata={"features": {"x": float(i)}, "label": i % 2})
            for i in range(8)
        ]
    )

    async def check():
        estimator = create_estimator(reference, **options)
        report = await estimator.fit(dataset)
        assert report.samples == 8
        before = await estimator.predict(dataset)
        assert len(before) == 8
        path = estimator.save(tmp_path / "model.json")
        restored = create_estimator(reference, **options).load(path)
        after = await restored.predict(dataset)
        assert [x.value for x in after] == [x.value for x in before]
        metrics = await restored.evaluate(dataset)
        assert metrics

    run_sync(check())
