"""Model creation: train, evaluate and persist ML models through aire.

aire orchestrates the ML ecosystem rather than reimplementing it: native
zero-dependency estimators work offline, and scikit-learn / PyTorch / Keras /
XGBoost / LightGBM / CatBoost / pandas / polars plug in lazily through the same
:class:`Estimator` contract, :class:`Transform` / :class:`Pipeline` stages, and
``backend:name`` refs.
"""

from aire.ml.callbacks import EarlyStopping, HistoryCallback
from aire.ml.compose import ColumnTransformer, FeatureUnion
from aire.ml.estimator import Estimator
from aire.ml.native import (
    CentroidClassifier,
    KNNClassifier,
    LinearRegressor,
    MajorityClassifier,
)
from aire.ml.pandas_bridge import (
    available_backends,
    dataset_to_frame,
    frame_to_dataset,
    predictions_to_frame,
)
from aire.ml.pipeline import Pipeline
from aire.ml.sklearn_adapter import SklearnEstimator
from aire.ml.torch_adapter import TorchEstimator
from aire.ml.transform import MinMaxScaler, StandardScaler, Transform, create_transform
from aire.ml.types import (
    FeatureVector,
    FitReport,
    Prediction,
    TaskType,
    extract_features,
    vectorize,
)

__all__ = [
    "CentroidClassifier",
    "ColumnTransformer",
    "EarlyStopping",
    "Estimator",
    "FeatureUnion",
    "FeatureVector",
    "FitReport",
    "HistoryCallback",
    "KNNClassifier",
    "LinearRegressor",
    "MajorityClassifier",
    "MinMaxScaler",
    "Pipeline",
    "Prediction",
    "SklearnEstimator",
    "StandardScaler",
    "TaskType",
    "TorchEstimator",
    "Transform",
    "available_backends",
    "create_transform",
    "dataset_to_frame",
    "extract_features",
    "frame_to_dataset",
    "predictions_to_frame",
    "vectorize",
]
