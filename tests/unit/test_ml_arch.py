"""Composable architecture blocks, optimizers, and losses."""

from __future__ import annotations

import pytest

from aire import AI
from aire.core.errors import NotFoundError
from aire.ml.arch import (
    compose,
)

torch = pytest.importorskip("torch", reason="aire[torch] required")


def test_available_blocks() -> None:
    avail = AI.ml.arch.available()
    for key in ("attention", "ffn", "norm", "residual", "embed", "head"):
        assert key in avail
        assert avail[key]
    assert "mha" in avail["attention"]
    assert "kda" in avail["attention"]
    assert "mla" in avail["attention"]
    assert "moe" in avail["ffn"]
    assert "latent_moe" in avail["ffn"]
    assert "swiglu" in avail["ffn"]
    assert "rmsnorm" in avail["norm"]
    assert "attn_res" in avail["residual"]


def test_construct_each_attention_and_ffn() -> None:
    x = torch.randn(2, 5, 32)
    for kind in AI.ml.arch.available()["attention"]:
        mod = AI.ml.arch.attention(kind, n_embd=32, n_head=4, block_size=32)
        out, cache = mod(x)
        assert out.shape == x.shape
        out2, _ = mod(x[:, :1], cache)
        assert out2.shape == (2, 1, 32)
    for kind in AI.ml.arch.available()["ffn"]:
        mod = AI.ml.arch.ffn(kind, n_embd=32, n_experts=4, top_k=2, n_shared=1)
        assert mod(x).shape == x.shape


def test_compose_heterogeneous_stack() -> None:
    model = AI.ml.arch.compose(
        layers=[
            {"attention": "mha", "ffn": "mlp", "norm": "rmsnorm"},
            {
                "attention": "delta",
                "ffn": "swiglu",
                "ffn_options": {"mult": 2},
            },
            {
                "attention": "mla",
                "ffn": "latent_moe",
                "attention_options": {"gated": True, "kv_rank": 16},
                "ffn_options": {"n_experts": 4, "top_k": 2, "n_shared": 1},
            },
            {"attention": "kda", "ffn": "situ_mlp"},
        ],
        n_embd=32,
        n_head=4,
        vocab_size=64,
        block_size=32,
        attn_res_every=2,
        name="hetero",
    )
    model.eval()
    idx = torch.randint(0, 64, (2, 6))
    logits, caches = model(idx)
    assert logits.shape == (2, 6, 64)
    assert len(caches) == 4
    info = model.describe()
    assert info["layers"][0]["attention"] == "mha"
    assert info["layers"][2]["attention"] == "mla"
    assert info["layers"][2]["ffn"] == "latent_moe"
    out = model.generate(idx[:1, :2], max_new_tokens=2)
    assert out.shape[1] == 4


def test_register_custom_attention_and_architecture() -> None:
    @AI.ml.arch.register_attention("scale_only", replace=True)
    def scale_only(*, n_embd: int, n_head: int = 1, **_: object) -> object:
        class Scale(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.kind = "scale_only"

            def forward(self, x: object, cache: object = None) -> tuple[object, object]:
                return x * self.scale, cache  # type: ignore[operator]

        return Scale()

    @AI.ml.arch.register_ffn("double", replace=True)
    def double_ffn(*, n_embd: int, **_: object) -> object:
        class Double(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kind = "double"

            def forward(self, x: object) -> object:
                return x  # type: ignore[return-value]

        return Double()

    @AI.ml.arch.register_architecture("toy_custom", replace=True)
    def toy_custom(**options: object) -> object:
        return compose(
            [{"attention": "scale_only", "ffn": "double"}],
            n_embd=16,
            n_head=1,
            vocab_size=32,
            block_size=16,
            name="toy_custom",
        )

    model = AI.ml.arch.create("toy_custom")
    logits, _ = model(torch.zeros(1, 3, dtype=torch.long))
    assert logits.shape == (1, 3, 32)
    assert "scale_only" in AI.ml.arch.available()["attention"]
    assert "toy_custom" in AI.ml.arch.available()["architecture"]


def test_recipe_architectures_are_compositions() -> None:
    mha = AI.ml.arch.create(
        "uniform_mha", n_layer=2, n_embd=32, n_head=4, vocab_size=64, block_size=32
    )
    hybrid = AI.ml.arch.create(
        "hybrid_cycle",
        n_layer=4,
        n_embd=32,
        n_head=4,
        vocab_size=64,
        block_size=32,
        cycle=["linear", "mha"],
        ffn_rest="mlp",
    )
    assert mha.describe()["layers"][0]["attention"] == "mha"
    assert hybrid.describe()["layers"][0]["attention"] == "linear"
    assert hybrid.describe()["layers"][1]["attention"] == "mha"


def test_unknown_block_raises() -> None:
    with pytest.raises(NotFoundError):
        AI.ml.arch.attention("not-a-real-attn", n_embd=32, n_head=4)


def test_optimizers_and_losses() -> None:
    model = AI.ml.arch.create(
        "uniform_mha", n_layer=1, n_embd=32, n_head=4, vocab_size=64, block_size=32
    )
    for name in ("sgd", "adam", "adamw", "rmsprop", "adagrad"):
        opt = AI.ml.optim.create(name, model.parameters(), lr=1e-3)
        assert opt is not None
    assert set(AI.ml.optim.available()) >= {"adam", "adamw", "sgd"}

    logits = torch.randn(4, 64, requires_grad=True)
    targets = torch.randint(0, 64, (4,))
    for name in ("cross_entropy", "mse", "l1", "huber", "bce", "kl_div", "moe_load_balance"):
        loss_fn = AI.ml.loss.create(name)
        if name == "cross_entropy":
            val = loss_fn(logits, targets)
            val.backward()
            assert val.ndim == 0
        elif name == "mse":
            val = loss_fn(logits, torch.zeros_like(logits))
            assert val.ndim == 0
        elif name == "moe_load_balance":
            probs = torch.softmax(torch.randn(4, 8), dim=-1)
            val = loss_fn(probs)
            assert val.ndim == 0
    assert "cross_entropy" in AI.ml.loss.available()
    assert AI.ml.optim.describe()["kind"] == "ml.optim"
    assert AI.ml.loss.describe()["kind"] == "ml.loss"


def test_ml_describe_mentions_arch_optim_loss() -> None:
    desc = AI.ml.describe()
    assert "arch" in desc
    assert "optim" in desc
    assert "loss" in desc
