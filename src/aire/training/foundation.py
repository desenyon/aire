"""Foundational model creation: GPT/LLaMA-style stacks, configs, and train hooks.

This builds a *toy architecture* (random-init / config-driven), NOT pretrained
weights. Do not treat ``create_foundation`` as loading GPT-2/LLaMA checkpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError

FoundationFamily = Literal["gpt2", "llama", "mistral", "moe", "custom"]
_FAMILIES = frozenset({"gpt2", "llama", "mistral", "moe", "custom"})


class FoundationConfig(BaseModel):
    """Declarative config for a small foundational LM stack."""

    family: FoundationFamily = "gpt2"
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    moe_experts: int = 4
    name: str = "foundation"
    extras: dict[str, Any] = Field(default_factory=dict)


class FoundationModel:
    """Composable foundational model handle (arch + optional trainer/quant).

    Toy stacks from :func:`create_foundation` have ``pretrained=False``.
    HF loads from :meth:`from_pretrained` set ``pretrained=True``.
    """

    def __init__(
        self,
        config: FoundationConfig,
        architecture: Any,
        *,
        pretrained: bool = False,
        hf_model_id: str | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.architecture = architecture
        self.pretrained = pretrained
        self.hf_model_id = hf_model_id
        self.tokenizer = tokenizer
        self._trainer: Any = None
        self._quantizer: Any = None

    def trainer(self, **options: Any) -> Any:
        from aire.training.lm_trainer import create_lm_trainer

        self._trainer = create_lm_trainer(self.architecture, **options)
        return self._trainer

    def quantize(self, **options: Any) -> Any:
        from aire.training.quantize import create_quantizer

        self._quantizer = create_quantizer(self.config.name, **options)
        return self._quantizer

    def lora(self, **options: Any) -> Any:
        from aire.training.lora import create_lora

        model_name = self.hf_model_id or self.config.name
        return create_lora(model_name, **options)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "gpt2",
        *,
        family: FoundationFamily | str | None = None,
        **options: Any,
    ) -> FoundationModel:
        """Load a real Hugging Face causal LM (requires ``aire[peft]`` / transformers).

        Unlike :func:`create_foundation`, this downloads pretrained weights.
        """
        import importlib.util

        if importlib.util.find_spec("transformers") is None:
            raise ConfigurationError(
                "transformers required for from_pretrained: pip install 'aire[peft]'",
                code="training.hf_missing",
                context={"extra": "aire[peft]", "model_id": model_id},
            )
        import transformers  # type: ignore[import-not-found]

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        cfg_src = getattr(model, "config", None)
        n_layer = int(
            getattr(cfg_src, "n_layer", None)
            or getattr(cfg_src, "num_hidden_layers", 0)
            or 0
        )
        n_embd = int(
            getattr(cfg_src, "n_embd", None) or getattr(cfg_src, "hidden_size", 0) or 0
        )
        n_head = int(
            getattr(cfg_src, "n_head", None)
            or getattr(cfg_src, "num_attention_heads", 0)
            or 0
        )
        vocab = int(getattr(cfg_src, "vocab_size", 0) or getattr(tokenizer, "vocab_size", 0) or 0)
        seq = int(
            getattr(cfg_src, "n_positions", None)
            or getattr(cfg_src, "max_position_embeddings", 1024)
            or 1024
        )
        fam: FoundationFamily = "custom"
        if family in _FAMILIES:
            fam = family  # type: ignore[assignment]
        elif "llama" in model_id.lower():
            fam = "llama"
        elif "mistral" in model_id.lower():
            fam = "mistral"
        elif "gpt2" in model_id.lower() or "gpt-2" in model_id.lower():
            fam = "gpt2"
        cfg = FoundationConfig(
            family=fam,
            n_layer=n_layer or 1,
            n_embd=n_embd or 1,
            n_head=n_head or 1,
            vocab_size=vocab or 1,
            max_seq_len=seq,
            name=model_id,
            extras={"source": "huggingface", **options},
        )
        return cls(cfg, model, pretrained=True, hf_model_id=model_id, tokenizer=tokenizer)

    def describe(self) -> dict[str, Any]:
        arch_desc = (
            self.architecture.describe()
            if hasattr(self.architecture, "describe")
            else {"type": type(self.architecture).__name__}
        )
        kind = (
            "foundation_pretrained"
            if self.pretrained
            else "foundation_toy_architecture"
        )
        honesty = (
            f"Hugging Face pretrained weights from {self.hf_model_id}"
            if self.pretrained
            else "NOT pretrained weights — random-init / config-driven toy stack"
        )
        return {
            "kind": kind,
            "honesty": honesty,
            "pretrained": self.pretrained,
            "hf_model_id": self.hf_model_id,
            "config": self.config.model_dump(),
            "architecture": arch_desc,
            "has_trainer": self._trainer is not None,
            "has_quantizer": self._quantizer is not None,
            "has_tokenizer": self.tokenizer is not None,
        }


def create_foundation(
    family: FoundationFamily | str = "gpt2",
    *,
    n_layer: int | None = None,
    n_embd: int | None = None,
    n_head: int | None = None,
    **options: Any,
) -> FoundationModel:
    """Build a toy foundational architecture from a family preset + overrides.

    This is NOT a pretrained model download — architecture/config only.
    """
    presets: dict[str, dict[str, Any]] = {
        "gpt2": {"n_layer": 4, "n_embd": 256, "n_head": 4, "attention": "mha", "ffn": "mlp"},
        "llama": {
            "n_layer": 8,
            "n_embd": 512,
            "n_head": 8,
            "attention": "mha",
            "ffn": "swiglu",
            "norm": "rmsnorm",
        },
        "mistral": {
            "n_layer": 8,
            "n_embd": 512,
            "n_head": 8,
            "attention": "mha",
            "ffn": "swiglu",
            "norm": "rmsnorm",
        },
        "moe": {
            "n_layer": 6,
            "n_embd": 384,
            "n_head": 6,
            "attention": "mha",
            "ffn": "moe",
            "moe_experts": 4,
        },
        "custom": {"n_layer": 4, "n_embd": 256, "n_head": 4, "attention": "mha", "ffn": "mlp"},
    }
    if family not in presets:
        raise ConfigurationError(
            f"unknown foundation family {family!r}",
            code="training.foundation_family",
            context={"available": sorted(presets)},
        )
    preset = dict(presets[family])
    if n_layer is not None:
        preset["n_layer"] = n_layer
    if n_embd is not None:
        preset["n_embd"] = n_embd
    if n_head is not None:
        preset["n_head"] = n_head
    preset.update(options)
    attention = str(preset.pop("attention", "mha"))
    ffn = str(preset.pop("ffn", "mlp"))
    norm = str(preset.pop("norm", "layernorm"))
    moe_experts = int(preset.pop("moe_experts", 4))
    layers = int(preset.pop("n_layer"))
    embd = int(preset.pop("n_embd", 256))
    heads = int(preset.pop("n_head", 4))
    vocab = int(preset.pop("vocab_size", 32000))
    seq = int(preset.pop("max_seq_len", 1024))
    dropout = float(preset.pop("dropout", 0.0))
    name = str(preset.pop("name", f"foundation-{family}"))
    family_lit: FoundationFamily = family if family in _FAMILIES else "custom"  # type: ignore[assignment]
    cfg = FoundationConfig(
        family=family_lit,
        n_layer=layers,
        n_embd=embd,
        n_head=heads,
        vocab_size=vocab,
        max_seq_len=seq,
        dropout=dropout,
        moe_experts=moe_experts,
        name=name,
        extras={"attention": attention, "ffn": ffn, "norm": norm, **preset},
    )
    from aire.ml.arch.compose import compose

    layer_specs: list[dict[str, Any]] = []
    for i in range(cfg.n_layer):
        use_ffn = ffn
        if family == "moe" and i == 0:
            use_ffn = "mlp"
        layer_specs.append({"attention": attention, "ffn": use_ffn, "norm": norm})
    compose_kwargs: dict[str, Any] = {
        "name": cfg.name,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.max_seq_len,
        "dropout": cfg.dropout,
    }
    if ffn == "moe":
        compose_kwargs["n_experts"] = moe_experts
    architecture = compose(layer_specs, **compose_kwargs)
    return FoundationModel(cfg, architecture)


def catalog() -> dict[str, Any]:
    return {
        "kind": "foundation_catalog",
        "families": sorted(_FAMILIES),
        "factory": "aire.training.foundation.create_foundation",
        "from_pretrained": "aire.training.foundation.FoundationModel.from_pretrained",
        "hooks": ["trainer", "quantize", "lora"],
        "honesty": "create_foundation=toy; from_pretrained=HF weights",
    }
