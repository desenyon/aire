"""0.2.6 hardening: richer ML metrics/CV/grid search + gateway spend/health/headers."""

from __future__ import annotations

from aire import AI
from aire.core.runtime import Runtime
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.deployment.gateway import create_gateway
from aire.ml.metrics import classification_report, regression_metrics
from tests.conftest import arun

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[misc, assignment]


def _clf_dataset() -> Dataset:
    rows = [
        ({"x": 0.0, "y": 0.0}, "a"),
        ({"x": 0.1, "y": 0.0}, "a"),
        ({"x": 0.0, "y": 0.1}, "a"),
        ({"x": 1.0, "y": 1.0}, "b"),
        ({"x": 1.1, "y": 1.0}, "b"),
        ({"x": 1.0, "y": 1.1}, "b"),
        ({"x": 0.05, "y": 0.05}, "a"),
        ({"x": 0.95, "y": 0.95}, "b"),
    ]
    return Dataset(
        name="toy",
        records=[
            Record(text=label, metadata={"features": feats, "label": label})
            for feats, label in rows
        ],
    )


def test_classification_report_and_evaluate() -> None:
    report = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert report.samples == 4
    assert 0.0 <= report.accuracy <= 1.0
    assert "a" in report.per_class
    metrics = report.as_metrics()
    assert "macro_f1" in metrics

    est = arun(AI.ml.fit("simple:centroid", _clf_dataset()))
    scored = arun(est.evaluate(_clf_dataset()))
    assert scored["accuracy"] >= 0.5
    assert "macro_f1" in scored


def test_regression_metrics_r2() -> None:
    m = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert m["r2"] == 1.0
    assert m["mae"] == 0.0


def test_cross_validate_and_grid_search() -> None:
    ds = _clf_dataset()
    cv = arun(AI.ml.cross_validate("simple:centroid", ds, k=4, seed=1))
    assert len(cv.folds) == 4
    assert "accuracy" in cv.mean

    gs = arun(
        AI.ml.grid_search(
            "simple:knn",
            ds,
            {"k": [1, 3]},
            k=2,
            seed=1,
        )
    )
    assert gs.best_params["k"] in (1, 3)
    assert len(gs.trials) == 2


def test_feature_importance_and_list() -> None:
    ds = _clf_dataset()
    est = arun(AI.ml.fit("simple:centroid", ds))
    imp = arun(est.feature_importance(ds, n_repeats=2, seed=2))
    assert set(imp) == {"x", "y"}
    catalog = AI.ml.catalog()
    assert any(r.startswith("simple:") for r in catalog["simple"])
    assert "torch:mlp" in catalog["torch"]


def test_gateway_health_spend_headers(runtime: Runtime) -> None:
    if TestClient is None:
        import pytest

        pytest.skip("fastapi missing")
    app = create_gateway(runtime, aliases={"echo": "mock:echo"}, budgets={"echo": 10.0})
    client = TestClient(app)
    h = client.get("/v1/health").json()
    assert h["status"] == "ok"
    assert "open_circuits" in h
    r = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert "X-Aire-Resolved-Model" in r.headers
    assert "X-Aire-Cost-Usd" in r.headers
    assert "X-Aire-Input-Tokens" in r.headers
    spend = client.get("/v1/gateway/spend").json()
    assert spend["day"]
    assert "echo" in spend["spend_usd"] or spend["budgets_usd"]["echo"] == 10.0
    assert "remaining_usd" in spend
