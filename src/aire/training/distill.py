"""Knowledge distillation adapter interfaces (experimental).

Teacher → student soft-target training without forcing a framework. Callers
supply tensors / logits; aire owns the contract, configs, and offline stub loss.

This is an experimental adapter — pure-Python KL over logit lists, not a
drop-in replacement for framework KD trainers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.training.trainer import TrainingConfig


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
            "experimental": True,
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


class DistillTrainer:
    """Orchestrate soft+hard distillation steps over paired logit batches.

    Offline-friendly: callers supply ``pairs`` of (student_logits, teacher_logits);
    loops through :class:`FunctionTrainer`.
    """

    def __init__(
        self,
        distiller: Distiller | None = None,
        *,
        config: TrainingConfig | None = None,
    ) -> None:
        self.distiller = distiller or Distiller()
        self.config = config or TrainingConfig(epochs=3)

    async def fit(
        self,
        pairs: list[tuple[list[float], list[float]]],
        *,
        epochs: int | None = None,
    ) -> Any:
        from aire.data.dataset import Dataset
        from aire.training.trainer import FunctionTrainer

        if not pairs:
            raise ConfigurationError(
                "DistillTrainer.fit requires non-empty (student, teacher) logit pairs",
                code="training.distill_empty",
            )
        cfg = self.config.model_copy(update={"epochs": epochs or self.config.epochs})

        def step(
            epoch: int,
            dataset: Dataset,
            config: TrainingConfig,
            state: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, Any]]:
            total = 0.0
            for student, teacher in pairs:
                total += self.distiller.step(student, teacher).total_loss
            mean = total / len(pairs)
            return {"loss": mean}, {"pairs": float(len(pairs))}

        trainer = FunctionTrainer(step, cfg)
        return await trainer.fit(Dataset.from_texts([f"pair-{i}" for i in range(len(pairs))]))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "distill_trainer",
            "experimental": True,
            "mode": "logit_pairs",
            "distiller": self.distiller.describe(),
            "config": self.config.model_dump(),
        }


class HFDistillTrainer:
    """End-to-end Hugging Face knowledge distillation (teacher → student).

    Requires ``transformers`` + ``torch``. Teacher is frozen; student is trained
    with temperature-scaled KL on vocabulary logits plus optional LM CE loss.
    """

    def __init__(
        self,
        student: str = "sshleifer/tiny-gpt2",
        teacher: str = "sshleifer/tiny-gpt2",
        *,
        config: DistillationConfig | None = None,
        output_dir: str = "./distill-out",
        dry_run: bool = False,
    ) -> None:
        self.student_id = student
        self.teacher_id = teacher
        self.config = config or DistillationConfig()
        self.output_dir = output_dir
        self.dry_run = dry_run
        self._student: Any = None
        self._teacher: Any = None
        self._tokenizer: Any = None

    def prepare(self) -> None:
        import importlib.util

        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec(
            "torch"
        ) is None:
            raise ConfigurationError(
                "HFDistillTrainer requires transformers+torch: pip install 'aire[peft]'",
                code="training.hf_distill_missing",
            )
        import torch
        import transformers  # type: ignore[import-not-found]

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.teacher_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._teacher = transformers.AutoModelForCausalLM.from_pretrained(self.teacher_id)
        self._student = transformers.AutoModelForCausalLM.from_pretrained(self.student_id)
        self._teacher.eval()
        for param in self._teacher.parameters():
            param.requires_grad = False
        self._torch = torch

    def fit(
        self,
        texts: list[str] | Dataset,
        *,
        epochs: int = 1,
        batch_size: int = 2,
        max_length: int = 64,
        learning_rate: float = 5e-5,
    ) -> dict[str, Any]:
        if isinstance(texts, Dataset):
            corpus = [r.text for r in texts if r.text.strip()]
        else:
            corpus = [t for t in texts if t.strip()]
        if not corpus:
            raise ConfigurationError(
                "HFDistillTrainer.fit requires non-empty texts",
                code="training.hf_distill_empty",
            )
        if self.dry_run:
            return {
                "kind": "hf_distill_result",
                "dry_run": True,
                "epochs": epochs,
                "samples": len(corpus),
                "student": self.student_id,
                "teacher": self.teacher_id,
            }
        if self._student is None:
            self.prepare()
        assert self._student is not None
        assert self._teacher is not None
        assert self._tokenizer is not None
        torch = self._torch
        optim = torch.optim.AdamW(self._student.parameters(), lr=learning_rate)
        history: list[dict[str, float]] = []
        temperature = max(self.config.temperature, 1e-6)
        alpha = self.config.alpha

        for epoch in range(epochs):
            total = 0.0
            n = 0
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i : i + batch_size]
                enc = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                input_ids = enc["input_ids"]
                attention = enc.get("attention_mask")
                with torch.no_grad():
                    t_out = self._teacher(input_ids=input_ids, attention_mask=attention)
                    t_logits = t_out.logits
                s_out = self._student(input_ids=input_ids, attention_mask=attention)
                s_logits = s_out.logits
                # Soft KD on next-token logits
                soft = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(s_logits / temperature, dim=-1),
                    torch.nn.functional.softmax(t_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                # Hard CE against teacher-forced labels
                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                hard = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self._tokenizer.pad_token_id or -100,
                )
                loss = alpha * soft + (1.0 - alpha) * hard
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += float(loss.detach())
                n += 1
            mean = total / max(n, 1)
            history.append({"epoch": float(epoch), "loss": mean})

        from pathlib import Path

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._student.save_pretrained(self.output_dir)
        self._tokenizer.save_pretrained(self.output_dir)
        return {
            "kind": "hf_distill_result",
            "dry_run": False,
            "epochs": epochs,
            "samples": len(corpus),
            "history": history,
            "output_dir": self.output_dir,
            "student": self.student_id,
            "teacher": self.teacher_id,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "hf_distill_trainer",
            "student": self.student_id,
            "teacher": self.teacher_id,
            "config": self.config.model_dump(),
            "dry_run": self.dry_run,
            "prepared": self._student is not None,
        }


def create_hf_distiller(
    student: str = "sshleifer/tiny-gpt2",
    teacher: str = "sshleifer/tiny-gpt2",
    **options: Any,
) -> HFDistillTrainer:
    return HFDistillTrainer(student, teacher, **options)
