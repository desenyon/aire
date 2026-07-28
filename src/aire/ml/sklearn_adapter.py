"""scikit-learn backend (``sklearn:<estimator>`` refs), lazily imported.

Short-name zoo + dotted-path escape hatch for *any* sklearn class.
Requires ``pip install aire[ml]``.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.ml.estimator import Estimator
from aire.ml.types import TaskType

_SKLEARN_NAMES: dict[str, tuple[str, str, TaskType]] = {
    # linear / glm
    "logistic_regression": ("sklearn.linear_model", "LogisticRegression", TaskType.CLASSIFICATION),
    "linear_regression": ("sklearn.linear_model", "LinearRegression", TaskType.REGRESSION),
    "ridge": ("sklearn.linear_model", "Ridge", TaskType.REGRESSION),
    "ridge_classifier": ("sklearn.linear_model", "RidgeClassifier", TaskType.CLASSIFICATION),
    "lasso": ("sklearn.linear_model", "Lasso", TaskType.REGRESSION),
    "elastic_net": ("sklearn.linear_model", "ElasticNet", TaskType.REGRESSION),
    "sgd_classifier": ("sklearn.linear_model", "SGDClassifier", TaskType.CLASSIFICATION),
    "sgd_regressor": ("sklearn.linear_model", "SGDRegressor", TaskType.REGRESSION),
    "perceptron": ("sklearn.linear_model", "Perceptron", TaskType.CLASSIFICATION),
    "passive_aggressive": (
        "sklearn.linear_model",
        "PassiveAggressiveClassifier",
        TaskType.CLASSIFICATION,
    ),
    "passive_aggressive_regressor": (
        "sklearn.linear_model",
        "PassiveAggressiveRegressor",
        TaskType.REGRESSION,
    ),
    "bayesian_ridge": ("sklearn.linear_model", "BayesianRidge", TaskType.REGRESSION),
    "huber": ("sklearn.linear_model", "HuberRegressor", TaskType.REGRESSION),
    "quantile": ("sklearn.linear_model", "QuantileRegressor", TaskType.REGRESSION),
    "poisson": ("sklearn.linear_model", "PoissonRegressor", TaskType.REGRESSION),
    # trees / ensembles
    "decision_tree": ("sklearn.tree", "DecisionTreeClassifier", TaskType.CLASSIFICATION),
    "decision_tree_regressor": ("sklearn.tree", "DecisionTreeRegressor", TaskType.REGRESSION),
    "extra_tree": ("sklearn.tree", "ExtraTreeClassifier", TaskType.CLASSIFICATION),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier", TaskType.CLASSIFICATION),
    "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor", TaskType.REGRESSION),
    "extra_trees": ("sklearn.ensemble", "ExtraTreesClassifier", TaskType.CLASSIFICATION),
    "extra_trees_regressor": ("sklearn.ensemble", "ExtraTreesRegressor", TaskType.REGRESSION),
    "gradient_boosting": (
        "sklearn.ensemble",
        "GradientBoostingClassifier",
        TaskType.CLASSIFICATION,
    ),
    "gradient_boosting_regressor": (
        "sklearn.ensemble",
        "GradientBoostingRegressor",
        TaskType.REGRESSION,
    ),
    "hist_gradient_boosting": (
        "sklearn.ensemble",
        "HistGradientBoostingClassifier",
        TaskType.CLASSIFICATION,
    ),
    "hist_gradient_boosting_regressor": (
        "sklearn.ensemble",
        "HistGradientBoostingRegressor",
        TaskType.REGRESSION,
    ),
    "ada_boost": ("sklearn.ensemble", "AdaBoostClassifier", TaskType.CLASSIFICATION),
    "ada_boost_regressor": ("sklearn.ensemble", "AdaBoostRegressor", TaskType.REGRESSION),
    "bagging": ("sklearn.ensemble", "BaggingClassifier", TaskType.CLASSIFICATION),
    "bagging_regressor": ("sklearn.ensemble", "BaggingRegressor", TaskType.REGRESSION),
    "voting_classifier": ("sklearn.ensemble", "VotingClassifier", TaskType.CLASSIFICATION),
    "stacking": ("sklearn.ensemble", "StackingClassifier", TaskType.CLASSIFICATION),
    "isolation_forest": ("sklearn.ensemble", "IsolationForest", TaskType.CLUSTERING),
    # svm / neighbors / nb / nn
    "svm": ("sklearn.svm", "SVC", TaskType.CLASSIFICATION),
    "svr": ("sklearn.svm", "SVR", TaskType.REGRESSION),
    "linear_svc": ("sklearn.svm", "LinearSVC", TaskType.CLASSIFICATION),
    "linear_svr": ("sklearn.svm", "LinearSVR", TaskType.REGRESSION),
    "nu_svc": ("sklearn.svm", "NuSVC", TaskType.CLASSIFICATION),
    "nu_svr": ("sklearn.svm", "NuSVR", TaskType.REGRESSION),
    "knn": ("sklearn.neighbors", "KNeighborsClassifier", TaskType.CLASSIFICATION),
    "knn_regressor": ("sklearn.neighbors", "KNeighborsRegressor", TaskType.REGRESSION),
    "radius_neighbors": (
        "sklearn.neighbors",
        "RadiusNeighborsClassifier",
        TaskType.CLASSIFICATION,
    ),
    "nearest_centroid": ("sklearn.neighbors", "NearestCentroid", TaskType.CLASSIFICATION),
    "lof": ("sklearn.neighbors", "LocalOutlierFactor", TaskType.CLUSTERING),
    "naive_bayes": ("sklearn.naive_bayes", "GaussianNB", TaskType.CLASSIFICATION),
    "bernoulli_nb": ("sklearn.naive_bayes", "BernoulliNB", TaskType.CLASSIFICATION),
    "multinomial_nb": ("sklearn.naive_bayes", "MultinomialNB", TaskType.CLASSIFICATION),
    "complement_nb": ("sklearn.naive_bayes", "ComplementNB", TaskType.CLASSIFICATION),
    "mlp": ("sklearn.neural_network", "MLPClassifier", TaskType.CLASSIFICATION),
    "mlp_regressor": ("sklearn.neural_network", "MLPRegressor", TaskType.REGRESSION),
    # discriminant / gp / isotonic / pls
    "lda": ("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis", TaskType.CLASSIFICATION),
    "qda": (
        "sklearn.discriminant_analysis",
        "QuadraticDiscriminantAnalysis",
        TaskType.CLASSIFICATION,
    ),
    "gaussian_process": (
        "sklearn.gaussian_process",
        "GaussianProcessClassifier",
        TaskType.CLASSIFICATION,
    ),
    "gaussian_process_regressor": (
        "sklearn.gaussian_process",
        "GaussianProcessRegressor",
        TaskType.REGRESSION,
    ),
    "isotonic": ("sklearn.isotonic", "IsotonicRegression", TaskType.REGRESSION),
    "pls": ("sklearn.cross_decomposition", "PLSRegression", TaskType.REGRESSION),
    # clustering
    "kmeans": ("sklearn.cluster", "KMeans", TaskType.CLUSTERING),
    "mini_batch_kmeans": ("sklearn.cluster", "MiniBatchKMeans", TaskType.CLUSTERING),
    "dbscan": ("sklearn.cluster", "DBSCAN", TaskType.CLUSTERING),
    "agglomerative": ("sklearn.cluster", "AgglomerativeClustering", TaskType.CLUSTERING),
    "spectral": ("sklearn.cluster", "SpectralClustering", TaskType.CLUSTERING),
    "birch": ("sklearn.cluster", "Birch", TaskType.CLUSTERING),
    "optics": ("sklearn.cluster", "OPTICS", TaskType.CLUSTERING),
    "mean_shift": ("sklearn.cluster", "MeanShift", TaskType.CLUSTERING),
    # calibration / multioutput
    "calibrated": (
        "sklearn.calibration",
        "CalibratedClassifierCV",
        TaskType.CLASSIFICATION,
    ),
    "multi_output_classifier": (
        "sklearn.multioutput",
        "MultiOutputClassifier",
        TaskType.MULTI_LABEL,
    ),
    "multi_output_regressor": (
        "sklearn.multioutput",
        "MultiOutputRegressor",
        TaskType.REGRESSION,
    ),
    "classifier_chain": ("sklearn.multioutput", "ClassifierChain", TaskType.MULTI_LABEL),
}

_SKLEARN_TRANSFORMS: dict[str, tuple[str, str]] = {
    "standard_scaler": ("sklearn.preprocessing", "StandardScaler"),
    "minmax_scaler": ("sklearn.preprocessing", "MinMaxScaler"),
    "maxabs_scaler": ("sklearn.preprocessing", "MaxAbsScaler"),
    "robust_scaler": ("sklearn.preprocessing", "RobustScaler"),
    "normalizer": ("sklearn.preprocessing", "Normalizer"),
    "binarizer": ("sklearn.preprocessing", "Binarizer"),
    "one_hot_encoder": ("sklearn.preprocessing", "OneHotEncoder"),
    "ordinal_encoder": ("sklearn.preprocessing", "OrdinalEncoder"),
    "label_encoder": ("sklearn.preprocessing", "LabelEncoder"),
    "label_binarizer": ("sklearn.preprocessing", "LabelBinarizer"),
    "polynomial_features": ("sklearn.preprocessing", "PolynomialFeatures"),
    "power_transformer": ("sklearn.preprocessing", "PowerTransformer"),
    "quantile_transformer": ("sklearn.preprocessing", "QuantileTransformer"),
    "spline_transformer": ("sklearn.preprocessing", "SplineTransformer"),
    "function_transformer": ("sklearn.preprocessing", "FunctionTransformer"),
    "pca": ("sklearn.decomposition", "PCA"),
    "truncated_svd": ("sklearn.decomposition", "TruncatedSVD"),
    "fast_ica": ("sklearn.decomposition", "FastICA"),
    "nmf": ("sklearn.decomposition", "NMF"),
    "kernel_pca": ("sklearn.decomposition", "KernelPCA"),
    "simple_imputer": ("sklearn.impute", "SimpleImputer"),
    "knn_imputer": ("sklearn.impute", "KNNImputer"),
    "missing_indicator": ("sklearn.impute", "MissingIndicator"),
    "select_k_best": ("sklearn.feature_selection", "SelectKBest"),
    "select_percentile": ("sklearn.feature_selection", "SelectPercentile"),
    "rfe": ("sklearn.feature_selection", "RFE"),
    "select_from_model": ("sklearn.feature_selection", "SelectFromModel"),
    "variance_threshold": ("sklearn.feature_selection", "VarianceThreshold"),
    "count_vectorizer": ("sklearn.feature_extraction.text", "CountVectorizer"),
    "tfidf_vectorizer": ("sklearn.feature_extraction.text", "TfidfVectorizer"),
    "hashing_vectorizer": ("sklearn.feature_extraction.text", "HashingVectorizer"),
    "dict_vectorizer": ("sklearn.feature_extraction", "DictVectorizer"),
    "feature_hasher": ("sklearn.feature_extraction", "FeatureHasher"),
}


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise ConfigurationError(
            "scikit-learn is required for sklearn:* estimators: pip install 'aire[ml]'",
            code="ml.sklearn_missing",
            context={"backend": "sklearn"},
        )


def _infer_task(class_name: str) -> TaskType:
    lower = class_name.lower()
    if any(k in lower for k in ("cluster", "outlier", "isolation", "lof", "optics")):
        return TaskType.CLUSTERING
    if "multioutput" in lower or "chain" in lower:
        return TaskType.MULTI_LABEL
    if "regressor" in lower or lower.endswith("regression"):
        return TaskType.REGRESSION
    return TaskType.CLASSIFICATION


def resolve_sklearn_class(name: str) -> tuple[type[Any], TaskType]:
    """Resolve a short name or dotted path to an estimator class + task."""
    _require_sklearn()
    if name in _SKLEARN_NAMES:
        module_name, class_name, task = _SKLEARN_NAMES[name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name), task
    if "." in name:
        module_name, _, class_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ConfigurationError(
                f"unknown sklearn estimator {name!r}",
                code="ml.estimator_unknown",
                context={"available": sorted(_SKLEARN_NAMES)},
                cause=exc,
            ) from exc
        return cls, _infer_task(class_name)
    raise ConfigurationError(
        f"unknown sklearn estimator {name!r}",
        code="ml.estimator_unknown",
        context={"available": sorted(_SKLEARN_NAMES)},
    )


def resolve_sklearn_transformer(name: str) -> type[Any]:
    """Resolve a short name or dotted path to a sklearn transformer class."""
    _require_sklearn()
    if name in _SKLEARN_TRANSFORMS:
        module_name, class_name = _SKLEARN_TRANSFORMS[name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)  # type: ignore[no-any-return]
    if "." in name:
        module_name, _, class_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)  # type: ignore[no-any-return]
        except (ImportError, AttributeError) as exc:
            raise ConfigurationError(
                f"unknown sklearn transformer {name!r}",
                code="ml.transform_unknown",
                context={"available": sorted(_SKLEARN_TRANSFORMS)},
                cause=exc,
            ) from exc
    raise ConfigurationError(
        f"unknown sklearn transformer {name!r}",
        code="ml.transform_unknown",
        context={"available": sorted(_SKLEARN_TRANSFORMS)},
    )


def sklearn_estimator_catalog() -> list[str]:
    return sorted(_SKLEARN_NAMES)


def sklearn_transform_catalog() -> list[str]:
    return sorted(_SKLEARN_TRANSFORMS)


class SklearnEstimator(Estimator):
    """Wraps any scikit-learn estimator behind the aire Estimator contract."""

    def __init__(self, name: str, **hyperparameters: Any) -> None:
        super().__init__()
        sample_weight = hyperparameters.pop("sample_weight", None)
        self._fit_sample_weight = sample_weight
        cls, task = resolve_sklearn_class(name)
        self.sklearn_name = name
        self.task = task
        self._model = cls(**hyperparameters)
        self._cluster_labels: list[int] = []

    def backend_name(self) -> str:
        return f"sklearn:{self.sklearn_name}"

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        import inspect

        params = inspect.signature(self._model.fit).parameters
        kwargs: dict[str, Any] = {}
        if self._fit_sample_weight is not None and "sample_weight" in params:
            kwargs["sample_weight"] = self._fit_sample_weight
        if self.task == TaskType.CLUSTERING:
            # unsupervised: ignore y when fit doesn't take it
            if "y" in params:
                try:
                    self._model.fit(x, y, **kwargs)
                except TypeError:
                    self._model.fit(x, **kwargs)
            else:
                self._model.fit(x, **kwargs)
            labels = getattr(self._model, "labels_", None)
            if labels is not None:
                self._cluster_labels = [int(v) for v in labels]
                n_clusters = len(set(self._cluster_labels) - {-1})
                return {"n_clusters": float(n_clusters), "samples": float(len(x))}
            return {"samples": float(len(x))}
        self._model.fit(x, y, **kwargs)
        if hasattr(self._model, "score"):
            try:
                score = float(self._model.score(x, y))
            except Exception:  # pragma: no cover - estimator variance
                score = 0.0
            metric = (
                "train_accuracy" if self.task == TaskType.CLASSIFICATION else "train_r2"
            )
            return {metric: score}
        return {"samples": float(len(x))}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        if hasattr(self._model, "predict"):
            predictions = self._model.predict(x)
            return [v.item() if hasattr(v, "item") else v for v in predictions]
        if hasattr(self._model, "fit_predict"):
            # DBSCAN-style at predict time isn't standard; return cluster labels if fitted
            raise ConfigurationError(
                f"{self.sklearn_name} has no predict(); use fit-time labels",
                code="ml.no_predict",
            )
        raise ConfigurationError(
            f"{self.sklearn_name} cannot predict", code="ml.no_predict"
        )

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if not hasattr(self._model, "predict_proba"):
            return None
        classes = [str(c) for c in self._model.classes_]
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            for row in self._model.predict_proba(x)
        ]

    def partial_fit(
        self, x: list[list[float]], y: list[Any], *, classes: list[Any] | None = None
    ) -> SklearnEstimator:
        """Incremental fit when the underlying estimator supports ``partial_fit``."""
        if not hasattr(self._model, "partial_fit"):
            raise ConfigurationError(
                f"{self.sklearn_name} does not support partial_fit",
                code="ml.no_partial_fit",
            )
        kwargs: dict[str, Any] = {}
        if classes is not None:
            kwargs["classes"] = classes
        self._model.partial_fit(x, y, **kwargs)
        if self.report is None:
            from aire.ml.types import FitReport

            self.report = FitReport(
                backend=self.backend_name(),
                task=str(self.task),
                samples=len(x),
                features=len(x[0]) if x else 0,
                metrics={"partial_fit": 1.0},
                feature_names=list(self.feature_names),
            )
        return self

    @property
    def model(self) -> Any:
        """The underlying sklearn estimator (persist it with skops/joblib)."""
        return self._model

    def native_feature_importances(self) -> dict[str, float] | None:
        """Tree/linear ``feature_importances_`` / ``coef_`` when available."""
        if not self.feature_names:
            return None
        if hasattr(self._model, "feature_importances_"):
            vals = [float(v) for v in self._model.feature_importances_]
            return dict(zip(self.feature_names, vals, strict=False))
        if hasattr(self._model, "coef_"):
            coef = self._model.coef_
            flat = coef[0] if hasattr(coef, "ndim") and coef.ndim > 1 else coef
            vals = [float(abs(v)) for v in flat]
            return dict(zip(self.feature_names, vals, strict=False))
        return None

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "sklearn models cannot be serialized by aire (no-pickle policy); "
            "persist estimator.model with skops.io or joblib instead",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "sklearn models cannot be deserialized by aire (no-pickle policy); "
            "load with skops.io or joblib instead",
            code="ml.persistence_delegated",
        )

    def describe(self) -> Manifest:
        manifest = super().describe()
        manifest.extra["hyperparameters"] = {
            k: v
            for k, v in self._model.get_params().items()
            if isinstance(v, (int, float, str, bool))
        }
        manifest.extra["partial_fit"] = hasattr(self._model, "partial_fit")
        return manifest


def register(runtime: Any) -> None:
    """Register the sklearn estimator factory on a runtime."""

    def _factory(name: str = "random_forest", *, runtime: Any = None, **options: Any) -> Estimator:
        return SklearnEstimator(name, **options)

    runtime.registry("estimator").register("sklearn", _factory, replace=True)
