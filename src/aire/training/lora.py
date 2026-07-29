"""PEFT / LoRA fine-tuning interface (lazy Hugging Face)."""

from __future__ import annotations

import importlib.util
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.training.trainer import Checkpoint, TrainResult


class LoRAConfig(BaseModel):
    """Minimal LoRA hyperparameters for PEFT adapters."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    dry_run: bool = Field(
        default=False,
        description="When True, LoRATrainer.fit validates data without HF Trainer.",
    )


def _require_peft() -> tuple[Any, Any]:
    if importlib.util.find_spec("peft") is None or importlib.util.find_spec("transformers") is None:
        raise ConfigurationError(
            "PEFT/transformers required for LoRA: pip install 'aire[peft]'",
            code="training.peft_missing",
            context={"extra": "aire[peft]", "packages": ["peft", "transformers"]},
        )
    import peft  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    return peft, transformers


def _texts_from(dataset: Dataset | list[str] | list[dict[str, Any]] | list[Any]) -> list[str]:
    if isinstance(dataset, Dataset):
        return [r.text for r in dataset if r.text.strip()]
    texts: list[str] = []
    for item in dataset:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(str(item.get("text") or item.get("content") or ""))
        else:
            texts.append(str(item))
    return [t for t in texts if t.strip()]


class LoRATrainer:
    """Wraps PEFT LoRA adaptation around a Hugging Face causal LM.

    Heavy imports are deferred until :meth:`prepare` / :meth:`fit` need them.
    Use ``config.dry_run=True`` (or ``fit(..., dry_run=True)``) for offline CI.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        config: LoRAConfig | None = None,
        output_dir: str = "./lora-out",
    ) -> None:
        self.model_name = model_name
        self.config = config or LoRAConfig()
        self.output_dir = output_dir
        self._model: Any = None
        self._tokenizer: Any = None
        self._peft_model: Any = None
        self._last_result: TrainResult | None = None

    def prepare(self) -> Any:
        """Load base model + apply LoRA adapters. Returns the PEFT model."""
        peft, transformers = _require_peft()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        lora = peft.LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            task_type=getattr(peft.TaskType, self.config.task_type, peft.TaskType.CAUSAL_LM),
        )
        self._peft_model = peft.get_peft_model(self._model, lora)
        return self._peft_model

    def fit(  # noqa: C901
        self,
        dataset: Dataset | list[str] | list[dict[str, Any]],
        *,
        epochs: int = 1,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        max_length: int = 128,
        dry_run: bool | None = None,
        resume_from_checkpoint: str | bool | None = None,
        save_strategy: str = "epoch",
        eval_ratio: float = 0.0,
    ) -> TrainResult:
        """Fine-tune with Hugging Face ``Trainer`` (or dry-run without GPU/PEFT).

        ``dry_run=True`` validates the dataset and returns a synthetic
        :class:`TrainResult` without loading weights — used for CI / recipes.

        ``resume_from_checkpoint``: path to a HF checkpoint dir, ``True`` to
        resume from ``output_dir``, or ``None`` for a fresh run.
        """
        texts = _texts_from(dataset)
        if not texts:
            raise ConfigurationError(
                "LoRA fit requires at least one non-empty text sample",
                code="training.lora_empty",
            )
        use_dry = self.config.dry_run if dry_run is None else dry_run
        if use_dry:
            result = TrainResult(
                epochs_completed=epochs,
                best_metric=0.0,
                history=[{"epoch": float(i), "loss": 1.0 / (i + 1)} for i in range(epochs)],
                stopped_early=False,
                checkpoints=[],
            )
            self._last_result = result
            return result

        _, transformers = _require_peft()
        if self._peft_model is None:
            self.prepare()
        if self._tokenizer is None or self._peft_model is None:
            raise ConfigurationError(
                "LoRA prepare() failed to load model/tokenizer",
                code="training.lora_not_prepared",
            )

        def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
            encoded: dict[str, Any] = self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return encoded

        eval_dataset: Any = None
        if importlib.util.find_spec("datasets") is not None:
            import datasets as hf_datasets  # type: ignore[import-not-found]

            hf_ds = hf_datasets.Dataset.from_dict({"text": texts})
            if 0.0 < eval_ratio < 1.0 and len(texts) >= 2:
                split = hf_ds.train_test_split(test_size=eval_ratio, seed=42)
                train_raw, eval_raw = split["train"], split["test"]
            else:
                train_raw, eval_raw = hf_ds, None
            tokenized = train_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
            tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})
            if eval_raw is not None:
                eval_dataset = eval_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
                eval_dataset = eval_dataset.map(lambda x: {"labels": x["input_ids"]})
        else:
            tokenized = _TorchTextDataset(texts, self._tokenizer, max_length=max_length)

        resume: str | bool | None = resume_from_checkpoint
        if resume is True:
            resume = self.output_dir

        training_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy=save_strategy,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = transformers.Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=tokenized,
            eval_dataset=eval_dataset,
        )
        train_out = trainer.train(resume_from_checkpoint=resume)
        metrics = getattr(train_out, "metrics", {}) or {}
        loss = float(metrics.get("train_loss", 0.0))
        float_history = {"epoch": float(epochs - 1), "loss": loss}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                float_history[str(key)] = float(value)
        result = TrainResult(
            epochs_completed=epochs,
            best_metric=loss,
            history=[float_history],
            stopped_early=False,
            checkpoints=[
                Checkpoint(epoch=epochs - 1, path=self.output_dir, metrics={"loss": loss})
            ],
        )
        self._last_result = result
        self.save()
        return result

    def resume(self, checkpoint: str | None = None, **fit_kwargs: Any) -> TrainResult:
        """Resume LoRA training from ``checkpoint`` (default: ``output_dir``)."""
        dataset = fit_kwargs.pop("dataset", None)
        if dataset is None:
            raise ConfigurationError(
                "resume() requires dataset=",
                code="training.lora_resume_dataset",
            )
        return self.fit(
            dataset,
            resume_from_checkpoint=checkpoint if checkpoint is not None else True,
            **fit_kwargs,
        )

    async def afit(
        self,
        dataset: Dataset | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> TrainResult:
        """Async wrapper around :meth:`fit` (HF Trainer is sync)."""
        import asyncio

        return await asyncio.to_thread(self.fit, dataset, **kwargs)

    def save(self, path: str | None = None) -> str:
        if self._peft_model is None:
            raise ConfigurationError(
                "call prepare() or fit() before save()",
                code="training.lora_not_prepared",
            )
        out = path or self.output_dir
        self._peft_model.save_pretrained(out)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(out)
        return out

    def describe(self) -> dict[str, Any]:
        available = (
            importlib.util.find_spec("peft") is not None
            and importlib.util.find_spec("transformers") is not None
        )
        return {
            "kind": "lora_trainer",
            "model": self.model_name,
            "available": available,
            "install": "pip install 'aire[peft]'",
            "config": self.config.model_dump(),
            "prepared": self._peft_model is not None,
            "methods": ["prepare", "fit", "afit", "resume", "save", "describe"],
            "dry_run": self.config.dry_run,
            "resume_supported": True,
            "last_result": self._last_result.model_dump() if self._last_result else None,
        }


class _TorchTextDataset:
    """Minimal map-style dataset when ``datasets`` extra is absent."""

    def __init__(self, texts: list[str], tokenizer: Any, *, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def create_lora(model_name: str = "gpt2", **options: Any) -> LoRATrainer:
    config = options.pop("config", None)
    if config is None and any(k in options for k in LoRAConfig.model_fields):
        cfg_keys = {k: options.pop(k) for k in list(options) if k in LoRAConfig.model_fields}
        config = LoRAConfig(**cfg_keys)
    return LoRATrainer(model_name, config=config, **options)
