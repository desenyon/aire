"""Plugin discovery and loading.

Plugins are ordinary Python packages that either:

1. expose entry points in the ``aire.providers`` / ``aire.plugins`` groups, or
2. are registered programmatically via ``plugins.register_module(...)``.

The contract a plugin must satisfy is documented in PLUGIN_SPEC.md; in short, a
plugin module exposes ``register(runtime) -> PluginInfo``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from aire.core.errors import PluginError

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

ENTRY_POINT_GROUPS = ("aire.plugins", "aire.providers")


class PluginInfo(BaseModel):
    """What a plugin declares about itself after registration."""

    name: str
    version: str = "0.0.0"
    provides: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)


@runtime_checkable
class PluginModule(Protocol):
    def register(self, runtime: Runtime) -> PluginInfo: ...


class PluginManager:
    """Discovers and activates plugins for a runtime."""

    def __init__(self) -> None:
        self._loaded: dict[str, PluginInfo] = {}

    @property
    def loaded(self) -> dict[str, PluginInfo]:
        return dict(self._loaded)

    def register_module(self, module: PluginModule, runtime: Runtime) -> PluginInfo:
        """Activate a plugin module that follows the register() contract."""
        name = getattr(module, "__name__", type(module).__name__)
        if name in self._loaded:
            return self._loaded[name]
        try:
            info = module.register(runtime)
        except PluginError:
            raise
        except Exception as exc:
            raise PluginError(
                f"plugin {name!r} failed to register: {exc}",
                code="plugin.register_failed",
                context={"plugin": name},
                cause=exc,
            ) from exc
        self._loaded[name] = info
        return info

    def load_entry_points(self, runtime: Runtime) -> list[PluginInfo]:
        """Discover and activate all installed plugins via entry points."""
        activated: list[PluginInfo] = []
        for group in ENTRY_POINT_GROUPS:
            for ep in metadata.entry_points(group=group):
                if ep.name in self._loaded:
                    continue
                try:
                    target: Any = ep.load()
                except Exception as exc:
                    raise PluginError(
                        f"failed to import plugin entry point {ep.name!r}: {exc}",
                        code="plugin.import_failed",
                        context={"entry_point": ep.name, "group": group},
                        cause=exc,
                    ) from exc
                activated.append(self._activate_entry_point(ep.name, target, runtime))
        return activated

    def _activate_entry_point(self, name: str, target: Any, runtime: Runtime) -> PluginInfo:
        register: Callable[[Runtime], PluginInfo] | None = getattr(target, "register", None)
        if callable(register):
            info = register(runtime)
        elif isinstance(target, PluginInfo):
            info = target
        else:
            raise PluginError(
                f"entry point {name!r} does not expose register(runtime) or PluginInfo",
                code="plugin.contract_violation",
                context={"entry_point": name},
            )
        self._loaded[name] = info
        return info

    def load_module(self, dotted_path: str, runtime: Runtime) -> PluginInfo:
        """Import and activate a plugin by dotted module path."""
        try:
            module = importlib.import_module(dotted_path)
        except ImportError as exc:
            raise PluginError(
                f"cannot import plugin module {dotted_path!r}: {exc}",
                code="plugin.import_failed",
                context={"module": dotted_path},
                cause=exc,
            ) from exc
        register = getattr(module, "register", None)
        if not callable(register):
            raise PluginError(
                f"module {dotted_path!r} has no register(runtime) function",
                code="plugin.contract_violation",
                context={"module": dotted_path},
            )
        info = cast("PluginInfo", register(runtime))
        self._loaded[dotted_path] = info
        return info

    def describe(self) -> dict[str, Any]:
        return {name: info.model_dump() for name, info in self._loaded.items()}
