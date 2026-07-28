"""PEFT / LoRA fine-tuning interface (lazy Hugging Face)."""

from __future__ import annotations

import importlib.util
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError


class LoRAConfig(BaseModel):
    """Minimal LoRA hyperparameters for PEFT adapters."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


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


class LoRATrainer:
    """Wraps PEFT LoRA adaptation around a Hugging Face causal LM.

    Heavy imports are deferred until :meth:`prepare` / :meth:`describe` needs them.
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

    def save(self, path: str | None = None) -> str:
        if self._peft_model is None:
            raise ConfigurationError(
                "call prepare() before save()",
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
        }


def create_lora(model_name: str = "gpt2", **options: Any) -> LoRATrainer:
    config = options.pop("config", None)
    if config is None and any(k in options for k in ("r", "lora_alpha", "target_modules")):
        cfg_keys = {k: options.pop(k) for k in list(options) if k in LoRAConfig.model_fields}
        config = LoRAConfig(**cfg_keys)
    return LoRATrainer(model_name, config=config, **options)
