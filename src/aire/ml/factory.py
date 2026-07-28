"""Estimator factory helpers — register all ML backends and resolve refs."""

from __future__ import annotations

from typing import Any

from aire.core.types import Ref


def ensure_estimators(runtime: Any) -> None:
    """Register every ML backend factory on ``runtime`` (idempotent)."""
    from aire.ml import boosting, keras_adapter, native, sklearn_adapter, torch_adapter

    if not runtime.estimators.has("simple"):
        native.register(runtime)
    if not runtime.estimators.has("sklearn"):
        sklearn_adapter.register(runtime)
    if not runtime.estimators.has("torch"):
        torch_adapter.register(runtime)
    if not runtime.estimators.has("keras"):
        keras_adapter.register(runtime)
    if not runtime.estimators.has("xgboost"):
        boosting.register_xgboost(runtime)
    if not runtime.estimators.has("lightgbm"):
        boosting.register_lightgbm(runtime)


def create_estimator(spec: str, *, runtime: Any | None = None, **options: Any) -> Any:
    """Create an estimator from a ``backend:name`` ref without going through ``AI``."""
    if runtime is None:
        from aire.ai import default_runtime

        runtime = default_runtime()
    ensure_estimators(runtime)
    ref = Ref.parse(spec if ":" in spec else f"simple:{spec}")
    return runtime.estimators.create(ref.provider, name=ref.name, runtime=runtime, **options)
