"""Registries for swappable architecture blocks and full architecture factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aire.core.errors import ConfigurationError, NotFoundError

Factory = Callable[..., Any]


class BlockRegistry:
    """Name → factory registry for one block kind (attention, ffn, …)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory] = {}

    def register(self, name: str, factory: Factory | None = None, *, replace: bool = False) -> Any:
        def _do(fn: Factory) -> Factory:
            if name in self._factories and not replace:
                raise ConfigurationError(
                    f"{self.kind} {name!r} already registered",
                    code="arch.duplicate",
                    context={"kind": self.kind, "name": name},
                )
            self._factories[name] = fn
            return fn

        if factory is not None:
            return _do(factory)
        return _do

    def create(self, name: str, **options: Any) -> Any:
        try:
            return self._factories[name](**options)
        except KeyError:
            raise NotFoundError(
                self.kind, name, context={"available": sorted(self._factories)}
            ) from None

    def names(self) -> list[str]:
        return sorted(self._factories)

    def has(self, name: str) -> bool:
        return name in self._factories

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "available": self.names()}


ATTENTION: BlockRegistry = BlockRegistry("attention")
FFN: BlockRegistry = BlockRegistry("ffn")
NORM: BlockRegistry = BlockRegistry("norm")
RESIDUAL: BlockRegistry = BlockRegistry("residual")
EMBED: BlockRegistry = BlockRegistry("embed")
HEAD: BlockRegistry = BlockRegistry("head")
ARCHITECTURE: BlockRegistry = BlockRegistry("architecture")


def ensure_builtins_registered() -> None:
    """Idempotent import of builtin block modules (registers factories)."""
    if ATTENTION.names():
        return
    from aire.ml.arch import (
        attention,
        embed,
        ffn,
        head,
        norm,
        residual,
    )

    _ = (attention, ffn, norm, residual, embed, head)
