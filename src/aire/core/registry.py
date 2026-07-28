"""Typed component registries.

Registries are the dependency-injection backbone: subsystems register factories
under string names, and callers resolve them at runtime. Combined with
:class:`~aire.core.types.Ref` (``provider:name``) this is how
``AI.models.use("openai:gpt-4o-mini")`` finds its provider without the core
importing any vendor code.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from aire.core.errors import NotFoundError, PluginError

T = TypeVar("T")

Factory = Callable[..., T]


class Registry(Generic[T]):
    """A thread-safe name → factory registry for one component kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory[T]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        factory: Factory[T] | None = None,
        *,
        replace: bool = False,
    ) -> Factory[T] | Callable[[Factory[T]], Factory[T]]:
        """Register a factory. Usable directly or as a decorator."""

        def _do_register(fn: Factory[T]) -> Factory[T]:
            with self._lock:
                if name in self._factories and not replace:
                    raise PluginError(
                        f"{self.kind} {name!r} is already registered",
                        code="registry.duplicate",
                        context={"kind": self.kind, "name": name},
                    )
                self._factories[name] = fn
            return fn

        if factory is not None:
            return _do_register(factory)
        return _do_register

    def unregister(self, name: str) -> None:
        with self._lock:
            self._factories.pop(name, None)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._factories

    def get_factory(self, name: str) -> Factory[T]:
        with self._lock:
            try:
                return self._factories[name]
            except KeyError:
                raise NotFoundError(
                    self.kind, name, context={"available": sorted(self._factories)}
                ) from None

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Instantiate the component registered under ``name``."""
        return self.get_factory(name)(*args, **kwargs)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "registered": self.names()}


class Registries:
    """Lazily-created collection of per-kind registries."""

    def __init__(self) -> None:
        self._registries: dict[str, Registry[Any]] = {}
        self._lock = threading.RLock()

    def of(self, kind: str) -> Registry[Any]:
        with self._lock:
            if kind not in self._registries:
                self._registries[kind] = Registry(kind)
            return self._registries[kind]

    def kinds(self) -> list[str]:
        with self._lock:
            return sorted(self._registries)

    def describe(self) -> dict[str, Any]:
        return {kind: reg.names() for kind, reg in self._registries.items()}
