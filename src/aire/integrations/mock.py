"""Plugin-compatible wrapper around the builtin mock provider.

Exposed as the ``mock``/``echo`` entry points in pyproject.toml so external
plugin discovery exercises the same registration path as real providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aire.core.plugins import PluginInfo

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


class MockProvider:
    """Entry-point target registering builtin offline providers."""

    @staticmethod
    def register(runtime: Runtime) -> PluginInfo:
        from aire.models.builtin import register_builtins

        register_builtins(runtime)
        return PluginInfo(
            name="mock",
            version="0.1.0",
            provides=["model:mock", "embedder:local", "model:callable"],
        )


EchoProvider = MockProvider


def register(runtime: Runtime) -> PluginInfo:
    return MockProvider.register(runtime)
