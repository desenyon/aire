"""Tool registry: discover, inspect and invoke tools by name."""

from __future__ import annotations

from typing import Any

from aire.core.errors import NotFoundError
from aire.tools.tool import Tool


class ToolRegistry:
    """A collection of named tools with manifest introspection."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = True) -> Tool:
        if tool.name in self._tools and not replace:
            from aire.core.errors import PluginError

            raise PluginError(
                f"tool {tool.name!r} already registered",
                code="registry.duplicate",
                context={"tool": tool.name},
            )
        self._tools[tool.name] = tool
        return tool

    def add(self, tool: Tool) -> Tool:
        """Alias for register()."""
        return self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise NotFoundError("tool", name, context={"available": self.names()}) from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[Any]:
        """Model-facing tool definitions for all registered tools."""
        return [t.definition() for t in self._tools.values()]

    def manifests(self) -> list[dict[str, Any]]:
        return [t.describe().model_dump(mode="json") for t in self._tools.values()]

    def __iter__(self) -> Any:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def describe(self) -> dict[str, Any]:
        return {"kind": "tool_registry", "tools": self.manifests()}
