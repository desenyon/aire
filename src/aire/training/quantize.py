"""Quantization adapter interfaces (lazy bitsandbytes / transformers).

These are agent-friendly contracts — prepare a model for lower-precision
inference without pulling heavy deps until :meth:`Quantizer.prepare` runs.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from aire.core.errors import ConfigurationError

QuantMethod = Literal["bitsandbytes", "gguf", "awq", "gptq", "stub"]


class QuantizationConfig(BaseModel):
    """Declarative quantization settings."""

    bits: Literal[4, 8] = 8
    method: QuantMethod = "bitsandbytes"
    load_in_4bit: bool = False
    load_in_8bit: bool = True
    compute_dtype: str = "float16"
    llm_int8_threshold: float = 6.0

    @model_validator(mode="after")
    def _sync_bit_flags(self) -> QuantizationConfig:
        if self.bits == 4:
            self.load_in_4bit = True
            self.load_in_8bit = False
        else:
            self.load_in_8bit = True
            self.load_in_4bit = False
        return self


class Quantizer:
    """Lazy quantization wrapper around Hugging Face causal LMs."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        config: QuantizationConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = config or QuantizationConfig()
        self._model: Any = None
        self._tokenizer: Any = None

    def prepare(self) -> Any:
        """Load a quantized model. Returns the model object (or a stub describe dict)."""
        if self.config.method == "stub":
            self._model = {"kind": "quantized_stub", "bits": self.config.bits}
            return self._model
        if self.config.method != "bitsandbytes":
            raise ConfigurationError(
                f"quantization method {self.config.method!r} is declared but not wired; "
                "use method='bitsandbytes' or method='stub' for offline describe()",
                code="training.quant_method",
                context={"method": self.config.method, "supported": ["bitsandbytes", "stub"]},
            )
        if importlib.util.find_spec("transformers") is None:
            raise ConfigurationError(
                "transformers required for quantization: pip install 'aire[peft]'",
                code="training.quant_missing",
                context={"extra": "aire[peft]", "packages": ["transformers", "bitsandbytes"]},
            )
        import transformers  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {}
        if self.config.load_in_4bit:
            kwargs["load_in_4bit"] = True
        if self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
        # bitsandbytes is optional; clear error if missing at load time
        if importlib.util.find_spec("bitsandbytes") is None and (
            self.config.load_in_4bit or self.config.load_in_8bit
        ):
            raise ConfigurationError(
                "bitsandbytes required for 4/8-bit load: pip install bitsandbytes",
                code="training.bitsandbytes_missing",
                context={"extra": "bitsandbytes"},
            )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        return self._model

    def describe(self) -> dict[str, Any]:
        bnb = importlib.util.find_spec("bitsandbytes") is not None
        transformers_ok = importlib.util.find_spec("transformers") is not None
        return {
            "kind": "quantizer",
            "model": self.model_name,
            "config": self.config.model_dump(),
            "prepared": self._model is not None,
            "available": {
                "transformers": transformers_ok,
                "bitsandbytes": bnb,
                "stub": True,
            },
            "install": "pip install 'aire[peft]' bitsandbytes",
        }


def create_quantizer(model_name: str = "gpt2", **options: Any) -> Quantizer:
    config = options.pop("config", None)
    if config is None and any(k in options for k in QuantizationConfig.model_fields):
        cfg = {k: options.pop(k) for k in list(options) if k in QuantizationConfig.model_fields}
        config = QuantizationConfig(**cfg)
    return Quantizer(model_name, config=config, **options)
