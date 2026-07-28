"""Knowledge distillation adapter interfaces.

Teacher → student soft-target training without forcing a framework. Callers
supply tensors / logits; aire owns the contract, configs, and offline stub loss.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from aire.core.errors import ConfigurationError


class DistillationConfig(BaseModel):
    """KD hyperparameters."""

    temperature: float = 2.0
    alpha: float = 0.5
    """Weight on soft (KL) loss; ``1 - alpha`` weights hard (task) loss."""
    reduction: str = "mean"


class DistillationResult(BaseModel):
    soft_loss: float
    hard_loss: float
    total_loss: float
    temperature: float
    alpha: float


def _softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [x / temperature for x in logits]
    m = max(scaled) if scaled else 0.0
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) with tiny epsilon for numerical stability."""
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q, strict=True))


def soft_kl_loss(
    student_logits: list[float],
    teacher_logits: list[float],
    *,
    temperature: float = 2.0,
) -> float:
    """Temperature-scaled KL between teacher and student distributions (pure Python)."""
    if len(student_logits) != len(teacher_logits):
        raise ConfigurationError(
            "student and teacher logits must have the same length",
            code="training.distill_shape",
            context={"student": len(student_logits), "teacher": len(teacher_logits)},
        )
    t = max(temperature, 1e-6)
    p = _softmax(teacher_logits, t)
    q = _softmax(student_logits, t)
    # Hinton: multiply KL by T^2 so gradient scale matches hard loss
    return _kl_divergence(p, q) * (t * t)


class Distiller:
    """Combine soft teacher loss with an optional hard task loss callable."""

    def __init__(
        self,
        *,
        config: DistillationConfig | None = None,
        hard_loss_fn: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self.config = config or DistillationConfig()
        self.hard_loss_fn = hard_loss_fn

    def step(
        self,
        student_logits: list[float],
        teacher_logits: list[float],
        *,
        hard_target: Any | None = None,
        student_pred: Any | None = None,
    ) -> DistillationResult:
        soft = soft_kl_loss(
            student_logits, teacher_logits, temperature=self.config.temperature
        )
        hard = 0.0
        if self.hard_loss_fn is not None and hard_target is not None:
            hard = float(self.hard_loss_fn(student_pred, hard_target))
        alpha = self.config.alpha
        total = alpha * soft + (1.0 - alpha) * hard
        return DistillationResult(
            soft_loss=soft,
            hard_loss=hard,
            total_loss=total,
            temperature=self.config.temperature,
            alpha=alpha,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "distiller",
            "config": self.config.model_dump(),
            "has_hard_loss": self.hard_loss_fn is not None,
        }


def create_distiller(**options: Any) -> Distiller:
    hard = options.pop("hard_loss_fn", None)
    config = options.pop("config", None)
    if config is None and any(k in options for k in DistillationConfig.model_fields):
        cfg = {k: options.pop(k) for k in list(options) if k in DistillationConfig.model_fields}
        config = DistillationConfig(**cfg)
    if options:
        raise ConfigurationError(
            f"unknown distiller options: {sorted(options)}",
            code="training.distill_options",
        )
    return Distiller(config=config, hard_loss_fn=hard)
