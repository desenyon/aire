"""Hyperparameter search over TrainingConfig / trainer factories.

Offline default: discrete + continuous random search (no Optuna required).
When ``aire[optuna]`` is installed, :func:`optuna_search` delegates to Optuna.
"""

from __future__ import annotations

import importlib.util
import math
import random
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError
from aire.training.trainer import TrainingConfig


class SearchSpace(BaseModel):
    """Mixed discrete / continuous search space."""

    discrete: dict[str, list[Any]] = Field(default_factory=dict)
    continuous: dict[str, tuple[float, float]] = Field(default_factory=dict)
    log_continuous: dict[str, tuple[float, float]] = Field(default_factory=dict)


class TrialResult(BaseModel):
    params: dict[str, Any]
    score: float
    trial: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HPOResult(BaseModel):
    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult] = Field(default_factory=list)
    direction: str = "minimize"
    backend: str = "random"

    def describe(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": len(self.trials),
            "direction": self.direction,
            "backend": self.backend,
        }


ObjectiveFn = Callable[[dict[str, Any]], float | Awaitable[float]]
TrainerFactory = Callable[..., Any]


def sample_params(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, choices in space.discrete.items():
        if not choices:
            continue
        params[key] = rng.choice(choices)
    for key, (low, high) in space.continuous.items():
        params[key] = rng.uniform(low, high)
    for key, (low, high) in space.log_continuous.items():
        if low <= 0 or high <= 0:
            raise ConfigurationError(
                f"log_continuous {key!r} bounds must be > 0",
                code="training.hpo_bounds",
            )
        params[key] = math.exp(rng.uniform(math.log(low), math.log(high)))
    return params


def apply_training_config(base: TrainingConfig | None, params: dict[str, Any]) -> TrainingConfig:
    """Merge sampled params into a TrainingConfig (known fields + hyperparameters)."""
    cfg = (base or TrainingConfig()).model_copy(deep=True)
    known = set(TrainingConfig.model_fields)
    hyper = dict(cfg.hyperparameters)
    for key, value in params.items():
        if key in known:
            setattr(cfg, key, value)
        else:
            hyper[key] = value
    cfg.hyperparameters = hyper
    return cfg


async def random_search(
    objective: ObjectiveFn,
    space: SearchSpace,
    *,
    n_trials: int = 10,
    direction: str = "minimize",
    seed: int = 0,
) -> HPOResult:
    """Random search over discrete/continuous spaces (fully offline)."""
    import inspect

    rng = random.Random(seed)
    trials: list[TrialResult] = []
    best_score = float("inf") if direction == "minimize" else float("-inf")
    best_params: dict[str, Any] = {}

    for i in range(n_trials):
        params = sample_params(space, rng)
        score = objective(params)
        if inspect.isawaitable(score):
            score = await score
        score_f = float(score)
        trials.append(TrialResult(params=params, score=score_f, trial=i))
        better = score_f < best_score if direction == "minimize" else score_f > best_score
        if better:
            best_score = score_f
            best_params = dict(params)

    return HPOResult(
        best_params=best_params,
        best_score=best_score,
        trials=trials,
        direction=direction,
        backend="random",
    )


async def search_trainer(
    factory: TrainerFactory,
    evaluate: Callable[[Any], float | Awaitable[float]],
    space: SearchSpace,
    *,
    base_config: TrainingConfig | None = None,
    n_trials: int = 10,
    direction: str = "minimize",
    seed: int = 0,
) -> HPOResult:
    """Sample TrainingConfig variants, build trainers via ``factory``, score them."""
    import inspect

    async def objective(params: dict[str, Any]) -> float:
        cfg = apply_training_config(base_config, params)
        extras = {k: v for k, v in params.items() if k not in cfg.model_fields}
        trainer = factory(config=cfg, **extras)
        score = evaluate(trainer)
        if inspect.isawaitable(score):
            score = await score
        return float(score)

    return await random_search(
        objective, space, n_trials=n_trials, direction=direction, seed=seed
    )


def optuna_search(
    objective: Callable[[Any], float],
    space: SearchSpace,
    *,
    n_trials: int = 10,
    direction: str = "minimize",
    seed: int = 0,
) -> HPOResult:
    """Optuna-backed search when installed; else ConfigurationError with hint."""
    if importlib.util.find_spec("optuna") is None:
        raise ConfigurationError(
            "optuna is required for optuna_search: pip install 'aire[optuna]' "
            "(or use random_search which needs no extras)",
            code="training.optuna_missing",
            context={"extra": "aire[optuna]", "fallback": "aire.training.hpo.random_search"},
        )
    import optuna  # type: ignore[import-not-found]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def _obj(trial: Any) -> float:
        params: dict[str, Any] = {}
        for key, choices in space.discrete.items():
            params[key] = trial.suggest_categorical(key, choices)
        for key, (low, high) in space.continuous.items():
            params[key] = trial.suggest_float(key, low, high)
        for key, (low, high) in space.log_continuous.items():
            params[key] = trial.suggest_float(key, low, high, log=True)
        return float(objective(params))

    study.optimize(_obj, n_trials=n_trials)
    trials = [
        TrialResult(params=dict(t.params), score=float(t.value or 0.0), trial=t.number)
        for t in study.trials
        if t.value is not None
    ]
    return HPOResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        trials=trials,
        direction=direction,
        backend="optuna",
    )


def describe() -> dict[str, Any]:
    return {
        "kind": "hpo",
        "backends": {
            "random": True,
            "optuna": importlib.util.find_spec("optuna") is not None,
        },
        "install_optuna": "pip install 'aire[optuna]'",
    }
