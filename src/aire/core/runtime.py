"""The runtime: one object wiring config, registries, plugins, events, resources.

``Runtime`` is the dependency-injection container for the whole library. The
top-level :class:`aire.AI` facade owns a runtime; advanced users can create
their own for isolation (tests, multi-tenant servers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.core.config import Settings
from aire.core.events import EventBus
from aire.core.lifecycle import ResourceManager
from aire.core.logging import get_logger
from aire.core.plugins import PluginManager
from aire.core.registry import Registries, Registry

if TYPE_CHECKING:
    from aire.observability.tracing import Tracer

logger = get_logger(__name__)


class Runtime:
    """Composition root for aire subsystems."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        discover_plugins: bool = False,
    ) -> None:
        self.settings = settings or Settings.load()
        self.registries = Registries()
        self.events = EventBus()
        self.resources = ResourceManager()
        self.plugins = PluginManager()
        self.tracer: Tracer | None = None
        self._closed = False
        _register_builtin_providers(self)
        if discover_plugins:
            self.plugins.load_entry_points(self)

    # -- registry access -----------------------------------------------------------

    def registry(self, kind: str) -> Registry[Any]:
        return self.registries.of(kind)

    @property
    def model_providers(self) -> Registry[Any]:
        return self.registries.of("model_provider")

    @property
    def embedders(self) -> Registry[Any]:
        return self.registries.of("embedder")

    @property
    def vector_stores(self) -> Registry[Any]:
        return self.registries.of("vector_store")

    @property
    def tools(self) -> Registry[Any]:
        return self.registries.of("tool")

    @property
    def metrics(self) -> Registry[Any]:
        return self.registries.of("metric")

    # -- lifecycle --------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.resources.aclose()
        self._closed = True
        self.events.emit("runtime.closed", source="runtime")

    async def __aenter__(self) -> Runtime:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # -- introspection ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Machine-readable inventory of everything registered — for agents."""
        return {
            "project": self.settings.project,
            "registries": self.registries.describe(),
            "plugins": list(self.plugins.loaded),
        }


def _register_builtin_providers(runtime: Runtime) -> None:
    """Register zero-dependency builtin providers (mock model, hashing embedder).

    Imports are local so the core package never pays import cost for
    integrations, and never depends on vendor SDKs.
    """
    from aire.models.builtin import register_builtins

    register_builtins(runtime)
