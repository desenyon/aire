"""Compose neural architectures from swappable blocks + optim/loss.

Requires: pip install 'aire[torch]'
"""

from __future__ import annotations

from aire import AI


def main() -> None:
    print("=== available blocks ===")
    for kind, names in AI.ml.arch.available().items():
        print(f"  {kind}: {', '.join(names)}")

    print("\n=== compose a heterogeneous stack ===")
    import torch

    model = AI.ml.arch.compose(
        layers=[
            {"attention": "mha", "ffn": "mlp"},
            {"attention": "linear", "ffn": "swiglu"},
            {
                "attention": "kda",
                "ffn": "moe",
                "ffn_options": {"n_experts": 4, "top_k": 2, "n_shared": 1},
            },
            {
                "attention": "mla",
                "ffn": "latent_moe",
                "attention_options": {"gated": True},
                "ffn_options": {"n_experts": 4, "n_shared": 1},
            },
        ],
        n_embd=32,
        n_head=4,
        vocab_size=64,
        block_size=32,
        attn_res_every=2,
    )
    print("  layers:", model.describe()["layers"])
    print("  params:", model.count_parameters())

    opt = AI.ml.optim.create("adamw", model.parameters(), lr=1e-3)
    loss_fn = AI.ml.loss.create("cross_entropy")
    idx = torch.randint(0, 64, (2, 8))
    logits, _ = model(idx[:, :-1])
    loss = loss_fn(logits.reshape(-1, 64), idx[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    print(f"  train step ok  loss={float(loss.detach()):.4f}")
    print("  optim:", AI.ml.optim.available())
    print("  loss:", AI.ml.loss.available())


if __name__ == "__main__":
    main()
