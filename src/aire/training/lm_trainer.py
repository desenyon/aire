"""Train ``arch.compose`` stacks with FunctionTrainer / toy causal LM loop.

Offline-capable: without torch, uses a pure-Python toy LM step. With
``aire[torch]`` / ``aire[training]``, runs a real causal LM cross-entropy loop.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.training.trainer import FunctionTrainer, TrainingConfig, TrainResult


def _require_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise ConfigurationError(
            "torch is required for LMTrainer GPU/CPU loops: pip install 'aire[training]'",
            code="training.torch_missing",
            context={"extra": "aire[training]", "fallback": "LMTrainer(backend='toy')"},
        )
    import torch

    return torch


class LMTrainer:
    """Causal LM trainer for composed architectures or a toy bag-of-tokens LM."""

    def __init__(
        self,
        architecture: Any | None = None,
        *,
        config: TrainingConfig | None = None,
        backend: str = "auto",
        vocab_size: int = 256,
    ) -> None:
        self.architecture = architecture
        self.config = config or TrainingConfig(strategy="self_supervised", backend="lm")
        self.backend = backend
        self.vocab_size = vocab_size
        self._resolved: str | None = None

    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.architecture is not None and importlib.util.find_spec("torch") is not None:
            return "torch"
        return "toy"

    async def fit(self, dataset: Dataset) -> TrainResult:
        backend = self._pick_backend()
        self._resolved = backend
        if backend == "torch":
            return await self._fit_torch(dataset)
        return await self._fit_toy(dataset)

    async def _fit_toy(self, dataset: Dataset) -> TrainResult:
        """Character-level frequency LM: loss decreases as unigram probs stabilize."""
        texts = [r.text or "" for r in dataset.records]
        counts = [0] * self.vocab_size
        total = 0

        def step(
            epoch: int,
            data: Dataset,
            config: TrainingConfig,
            state: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, Any]]:
            nonlocal total
            for text in texts:
                for ch in text:
                    counts[ord(ch) % self.vocab_size] += 1
                    total += 1
            # Negative log-likelihood under empirical unigram (toy).
            loss = 0.0
            n = 0
            for text in texts:
                for ch in text:
                    p = counts[ord(ch) % self.vocab_size] / max(total, 1)
                    loss -= __import__("math").log(max(p, 1e-9))
                    n += 1
            avg = loss / max(n, 1)
            state["counts"] = list(counts)
            state["total"] = total
            return {"loss": avg, "tokens": float(n)}, state

        trainer = FunctionTrainer(step, self.config, metric="loss", minimize=True)
        return await trainer.fit(dataset)

    async def _fit_torch(self, dataset: Dataset) -> TrainResult:
        torch = _require_torch()
        model = self.architecture
        if model is None:
            raise ConfigurationError(
                "architecture required for torch LM backend "
                "(pass AI.ml.arch.compose(...) or set backend='toy')",
                code="training.lm_arch_missing",
            )
        if not hasattr(model, "parameters"):
            raise ConfigurationError(
                "architecture must be a torch nn.Module (or expose .parameters())",
                code="training.lm_arch_invalid",
            )
        model.train()
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )

        def _encode(text: str) -> Any:
            ids = [ord(c) % self.vocab_size for c in (text or " ")][:64]
            if len(ids) < 2:
                ids = [0, 1]
            return torch.tensor(ids, dtype=torch.long)

        def step(
            epoch: int,
            data: Dataset,
            config: TrainingConfig,
            state: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, Any]]:
            total_loss = 0.0
            n = 0
            for record in data.records:
                tokens = _encode(record.text or "")
                inputs = tokens[:-1].unsqueeze(0)
                targets = tokens[1:].unsqueeze(0)
                optim.zero_grad()
                try:
                    logits = model(inputs)
                except TypeError:
                    # Some composed stacks expect different call signatures.
                    logits = model(inputs, targets)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.dim() == 3:
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                else:
                    loss = logits.mean()  # fallback
                loss.backward()
                optim.step()
                total_loss += float(loss.detach())
                n += 1
            return {"loss": total_loss / max(n, 1)}, state

        trainer = FunctionTrainer(step, self.config, metric="loss", minimize=True)
        return await trainer.fit(dataset)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "lm_trainer",
            "backend": self._resolved or self.backend,
            "torch_available": importlib.util.find_spec("torch") is not None,
            "config": self.config.model_dump(),
            "has_architecture": self.architecture is not None,
        }


def create_lm_trainer(architecture: Any | None = None, **options: Any) -> LMTrainer:
    return LMTrainer(architecture, **options)
