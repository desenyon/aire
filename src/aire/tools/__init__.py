"""Tool system: self-describing, permissioned, auditable callables."""

from aire.tools.builtins import builtin_tools
from aire.tools.openapi import load_openapi, openapi_to_tools
from aire.tools.registry import ToolRegistry
from aire.tools.tool import Tool, tool
from aire.tools.types import RetryPolicy, SideEffect, ToolResult, ToolSpec

__all__ = [
    "RetryPolicy",
    "SideEffect",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "builtin_tools",
    "load_openapi",
    "openapi_to_tools",
    "tool",
]
