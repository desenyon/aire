"""Pipeline: chain Transforms then a final Estimator under one aire contract."""

from __future__ import annotations

from typing import Any, cast

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.estimator import Estimator
from aire.ml.transform import Transform, create_transform
from aire.ml.types import FitReport, Prediction, extract_features, vectorize


class Pipeline(Estimator):
    """Compose ``[transform, ...] → estimator`` with a single fit/predict API.

    Example::

        pipe = Pipeline(
            steps=[("scale", "native:standard_scaler"), ("clf", "simple:centroid")]
        )
        # or pass live objects:
        pipe = Pipeline(steps=[("scale", StandardScaler()), ("clf", CentroidClassifier())])
    """

    def __init__(
        self,
        steps: list[tuple[str, str | Transform | Estimator]],
        *,
        target: str = "label",
    ) -> None:
        super().__init__()
        if not steps:
            raise ConfigurationError("pipeline needs at least one step", code="ml.empty_pipeline")
        self.target_field = target
        self.step_names = [name for name, _ in steps]
        self.transforms: list[Transform] = []
        self.estimator: Estimator | None = None
        for i, (name, step) in enumerate(steps):
            is_last = i == len(steps) - 1
            resolved = self._resolve_step(step, final=is_last)
            if isinstance(resolved, Estimator):
                if not is_last:
                    raise ConfigurationError(
                        f"estimator step {name!r} must be last in the pipeline",
                        code="ml.pipeline_order",
                    )
                self.estimator = resolved
                self.task = resolved.task
            else:
                self.transforms.append(resolved)
        if self.estimator is None:
            raise ConfigurationError(
                "pipeline must end with an Estimator",
                code="ml.pipeline_no_estimator",
            )

    @staticmethod
    def _resolve_step(
        step: str | Transform | Estimator, *, final: bool
    ) -> Transform | Estimator:
        if isinstance(step, (Transform, Estimator)):
            return step
        if not isinstance(step, str):
            raise ConfigurationError(
                f"pipeline step must be str|Transform|Estimator, got {type(step)}",
                code="ml.pipeline_step",
            )
        # Heuristic: estimator refs look like provider:name where provider is known
        provider = step.split(":", 1)[0] if ":" in step else "native"
        native_estimators = {"majority", "centroid", "knn", "linear_regression"}
        if provider in {
            "simple",
            "sklearn",
            "torch",
            "keras",
            "xgboost",
            "lightgbm",
            "catboost",
        } or (final and step.split(":")[-1] in native_estimators):
            from aire.ml.factory import create_estimator

            ref = step if ":" in step else f"simple:{step}"
            if provider == "native" and ":" in step:
                ref = f"simple:{step.split(':', 1)[1]}"
            return cast(Estimator, create_estimator(ref))
        return create_transform(step)

    def backend_name(self) -> str:
        est = self.estimator.backend_name() if self.estimator else "none"
        return f"pipeline[{'+'.join(self.step_names)} → {est}]"

    def _prepare_xy(
        self, dataset: Dataset, target: str
    ) -> tuple[list[list[float]], list[Any], list[str]]:
        rows = [extract_features(record) for record in dataset]
        names, x = vectorize(rows)
        y = []
        for record in dataset:
            if target not in record.metadata:
                raise ConfigurationError(
                    f"record {record.id} missing target {target!r}",
                    code="ml.target_missing",
                )
            y.append(record.metadata[target])
        self.feature_names = names
        return x, y, names

    async def fit(self, dataset: Dataset, *, target: str | None = None) -> FitReport:
        import time

        started = time.time()
        target = target or self.target_field
        x, y, names = self._prepare_xy(dataset, target)
        self.feature_names = names
        for transform in self.transforms:
            transform.feature_names = list(self.feature_names)
            x = transform.fit_transform_matrix(x, y)
        assert self.estimator is not None
        # After transforms, feature dim may change — keep names only if width matches
        width = len(x[0]) if x else 0
        feat_names = names if width == len(names) else [f"f{i}" for i in range(width)]
        records = [
            Record(
                id=f"pipe-{i}",
                text="",
                metadata={
                    "features": dict(zip(feat_names, row, strict=True)),
                    target: label,
                },
            )
            for i, (row, label) in enumerate(zip(x, y, strict=True))
        ]
        self._out_features = feat_names
        report = await self.estimator.fit(
            Dataset(name="pipeline-fit", records=records), target=target
        )
        self.task = self.estimator.task
        self.report = FitReport(
            backend=self.backend_name(),
            task=str(self.task),
            samples=report.samples,
            features=len(names),
            metrics=dict(report.metrics),
            feature_names=list(names),
            duration_s=time.time() - started,
        )
        return self.report

    async def predict(self, inputs: Dataset | list[Record]) -> list[Prediction]:
        if self.report is None or self.estimator is None:
            raise ConfigurationError("pipeline is not fitted", code="ml.not_fitted")
        records = list(inputs if isinstance(inputs, list) else list(inputs))
        rows = [extract_features(record) for record in records]
        x = [[row.get(name, 0.0) for name in self.feature_names] for row in rows]
        for transform in self.transforms:
            x = transform.transform_matrix(x)
        feat_names = getattr(self, "_out_features", [f"f{i}" for i in range(len(x[0]) if x else 0)])
        transformed = [
            Record(
                id=record.id,
                text=record.text,
                metadata={"features": dict(zip(feat_names, row, strict=True))},
            )
            for record, row in zip(records, x, strict=True)
        ]
        return await self.estimator.predict(transformed)

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        raise ConfigurationError("use Pipeline.fit(dataset)", code="ml.use_async_fit")

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        raise ConfigurationError("use Pipeline.predict(records)", code="ml.use_async_predict")

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "pipeline persistence: save each transform/estimator separately",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "pipeline persistence: load each transform/estimator separately",
            code="ml.persistence_delegated",
        )

    def describe(self) -> Manifest:
        manifest = super().describe()
        manifest.extra.update(
            {
                "steps": self.step_names,
                "transforms": [t.name for t in self.transforms],
                "estimator": self.estimator.backend_name() if self.estimator else None,
            }
        )
        return manifest
