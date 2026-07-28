"""Stack an arbitrary sequence of decoder blocks into a language model."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.block import block_from_spec
from aire.ml.arch.registry import EMBED, HEAD, RESIDUAL, ensure_builtins_registered
from aire.ml.arch.specs import LayerSpec, StackSpec
from aire.ml.arch.torch_util import require_torch


def build_stack(spec: StackSpec) -> Any:  # noqa: C901
    """Materialize a torch module from a fully custom :class:`StackSpec`."""
    ensure_builtins_registered()
    torch = require_torch()
    cfg = spec

    class ComposedLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = cfg
            self.embed = EMBED.create(
                cfg.embed,
                vocab_size=cfg.vocab_size,
                n_embd=cfg.n_embd,
                block_size=cfg.block_size,
                dropout=cfg.dropout,
                **cfg.embed_options,
            )
            self.blocks = torch.nn.ModuleList(
                [
                    block_from_spec(
                        layer,
                        n_embd=cfg.n_embd,
                        n_head=cfg.n_head,
                        dropout=cfg.dropout,
                        bias=cfg.bias,
                        block_size=cfg.block_size,
                    )
                    for layer in cfg.layers
                ]
            )
            from aire.ml.arch.registry import NORM

            self.ln_f = NORM.create("layernorm", n_embd=cfg.n_embd)
            self.lm_head = HEAD.create(
                "lm" if cfg.head == "tied" else cfg.head,
                vocab_size=cfg.vocab_size,
                n_embd=cfg.n_embd,
                **cfg.head_options,
            )
            if cfg.head == "tied" and hasattr(self.embed, "wte"):
                self.lm_head.weight = self.embed.wte.weight
            self.attn_res = None
            self.attn_res_every = cfg.attn_res_every
            if cfg.attn_res_every:
                self.attn_res = RESIDUAL.create("attn_res", n_embd=cfg.n_embd)

        def forward(
            self, idx: Any, *, caches: list[Any] | None = None, start: int = 0
        ) -> tuple[Any, list[Any]]:
            x = self.embed(idx, start=start)
            new_caches: list[Any] = []
            depth_blocks: list[Any] = []
            every = self.attn_res_every
            for i, block in enumerate(self.blocks):
                cache_i = caches[i] if caches is not None else None
                x, cache_out = block(x, cache_i)
                new_caches.append(cache_out)
                if every and self.attn_res is not None and (i + 1) % every == 0:
                    depth_blocks.append(x.clone())
                    x = self.attn_res.remix(depth_blocks, None)
            if (
                self.attn_res is not None
                and depth_blocks
                and every is not None
                and len(self.blocks) % every != 0
            ):
                x = self.attn_res.remix(depth_blocks, x)
            return self.lm_head(self.ln_f(x)), new_caches

        @torch.no_grad()
        def generate(self, idx: Any, *, max_new_tokens: int = 16, temperature: float = 1.0) -> Any:
            logits, caches = self.forward(idx, caches=None, start=0)
            start = idx.size(1)
            out = idx
            for _ in range(max_new_tokens):
                nxt = torch.multinomial(
                    torch.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1), 1
                )
                out = torch.cat([out, nxt], dim=1)
                if start >= self.spec.block_size:
                    break
                logits, caches = self.forward(nxt, caches=caches, start=start)
                start += 1
            return out

        def count_parameters(self, *, trainable_only: bool = True) -> int:
            params = (
                (p for p in self.parameters() if p.requires_grad)
                if trainable_only
                else self.parameters()
            )
            return sum(p.numel() for p in params)

        def describe(self) -> dict[str, Any]:
            return {
                "kind": "composed_lm",
                "spec": self.spec.describe(),
                "parameters": self.count_parameters(),
                "layers": [
                    {
                        "index": i,
                        "attention": b.attention_kind,
                        "ffn": b.ffn_kind,
                        "norm": b.norm_kind,
                        "residual": b.residual_kind,
                    }
                    for i, b in enumerate(self.blocks)
                ],
                "capabilities": ["forward", "generate", "describe", "swap_blocks"],
            }

    return ComposedLM()


def layers_from_dicts(raw: list[dict[str, Any]]) -> list[LayerSpec]:
    return [LayerSpec.model_validate(item) for item in raw]
