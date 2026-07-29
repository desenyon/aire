"""Train ``arch.compose`` stacks with FunctionTrainer / causal LM loop.

Offline-capable: without torch, uses a pure-Python toy LM step. With
``aire[torch]`` / ``aire[training]``, runs a batched causal LM cross-entropy
loop with optional eval split and checkpoint directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.training.trainer import Checkpoint, FunctionTrainer, TrainingConfig, TrainResult


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
        batch_size: int = 4,
        max_length: int = 64,
        eval_ratio: float = 0.0,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.architecture = architecture
        self.config = config or TrainingConfig(strategy="self_supervised", backend="lm")
        self.backend = backend
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_length = max_length
        self.eval_ratio = eval_ratio
        self.checkpoint_dir = checkpoint_dir
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

    def _split(self, records: list[Any]) -> tuple[list[Any], list[Any]]:
        if not (0.0 < self.eval_ratio < 1.0) or len(records) < 2:
            return records, []
        cut = max(1, int(len(records) * (1.0 - self.eval_ratio)))
        return records[:cut], records[cut:]

    async def _fit_torch(self, dataset: Dataset) -> TrainResult:  # noqa: C901
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
        optim = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        train_recs, eval_recs = self._split(list(dataset.records))
        ckpt_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        def _encode(text: str) -> Any:
            ids = [ord(c) % self.vocab_size for c in (text or " ")][: self.max_length]
            if len(ids) < 2:
                ids = [0, 1]
            return torch.tensor(ids, dtype=torch.long)

        def _batch_loss(recs: list[Any]) -> float:
            total_loss = 0.0
            n = 0
            for i in range(0, len(recs), self.batch_size):
                chunk = recs[i : i + self.batch_size]
                batch_loss: Any = 0.0
                for record in chunk:
                    tokens = _encode(record.text or "")
                    inputs = tokens[:-1].unsqueeze(0)
                    targets = tokens[1:].unsqueeze(0)
                    try:
                        logits = model(inputs)
                    except TypeError:
                        logits = model(inputs, targets)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if logits.dim() == 3:
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1),
                        )
                    else:
                        loss = logits.mean()
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / max(len(chunk), 1)
                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                total_loss += float(batch_loss.detach())
                n += 1
            return total_loss / max(n, 1)

        def step(
            epoch: int,
            data: Dataset,
            config: TrainingConfig,
            state: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, Any]]:
            train_loss = _batch_loss(train_recs)
            metrics: dict[str, float] = {"loss": train_loss, "train_loss": train_loss}
            if eval_recs:
                model.eval()
                with torch.no_grad():
                    eval_loss = 0.0
                    m = 0
                    for record in eval_recs:
                        tokens = _encode(record.text or "")
                        inputs = tokens[:-1].unsqueeze(0)
                        targets = tokens[1:].unsqueeze(0)
                        try:
                            logits = model(inputs)
                        except TypeError:
                            logits = model(inputs, targets)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        if logits.dim() == 3:
                            loss = torch.nn.functional.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1),
                            )
                        else:
                            loss = logits.mean()
                        eval_loss += float(loss)
                        m += 1
                model.train()
                metrics["eval_loss"] = eval_loss / max(m, 1)
            if ckpt_dir is not None and hasattr(model, "state_dict"):
                path = ckpt_dir / f"epoch-{epoch}.pt"
                torch.save(model.state_dict(), path)
                state.setdefault("checkpoints", []).append(str(path))
            return metrics, state

        trainer = FunctionTrainer(step, self.config, metric="loss", minimize=True)
        result = await trainer.fit(dataset)
        if ckpt_dir is not None:
            # Attach checkpoint paths into result if TrainResult supports it
            ckpts = [
                Checkpoint(epoch=i, path=str(ckpt_dir / f"epoch-{i}.pt"), metrics={})
                for i in range(result.epochs_completed)
            ]
            return result.model_copy(update={"checkpoints": ckpts})
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "lm_trainer",
            "backend": self._resolved or self.backend,
            "torch_available": importlib.util.find_spec("torch") is not None,
            "config": self.config.model_dump(),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "eval_ratio": self.eval_ratio,
            "checkpoint_dir": self.checkpoint_dir,
            "has_architecture": self.architecture is not None,
        }


def create_lm_trainer(architecture: Any | None = None, **options: Any) -> LMTrainer:
    return LMTrainer(architecture, **options)
