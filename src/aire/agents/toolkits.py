"""Named toolkits — groups of tools for common agent roles."""

from __future__ import annotations

import ast
import json
import math
import operator
import statistics
from pathlib import Path
from typing import Any

from aire.core.errors import ConfigurationError, ToolError
from aire.tools.builtins import builtin_tools
from aire.tools.tool import Tool
from aire.tools.types import SideEffect

_SAFE_OPS: dict[type[Any], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right)))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(n.op)](_eval(n.operand)))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == "abs" and len(n.args) == 1:
                return abs(_eval(n.args[0]))
            if n.func.id == "round" and len(n.args) == 1:
                return float(round(_eval(n.args[0])))
        raise ToolError(f"unsafe expression: {expr!r}")

    return _eval(node)


def web_toolkit() -> list[Tool]:
    """HTTP-oriented tools from builtins (when present) plus a URL join helper."""
    tools: list[Tool] = []
    by_name = {t.spec.name: t for t in builtin_tools()}
    for name in ("http_get", "http_post", "web_search"):
        if name in by_name:
            tools.append(by_name[name])

    async def url_join(base: str, path: str) -> str:
        return base.rstrip("/") + "/" + path.lstrip("/")

    url_join.__doc__ = "Join a base URL and path safely."
    tools.append(Tool(url_join, name="url_join", side_effect=SideEffect.READ_ONLY))
    return tools


def code_toolkit() -> list[Tool]:
    """Safe local coding helpers (no arbitrary exec).

    Prefers the shared ``calculator`` from builtins when available.
    """
    tools: list[Tool] = []
    by_name = {t.spec.name: t for t in builtin_tools()}
    if "calculator" in by_name:
        tools.append(by_name["calculator"])
    else:

        async def calc(expression: str) -> str:
            return str(_safe_eval(expression))

        calc.__doc__ = "Evaluate a safe arithmetic expression."
        tools.append(Tool(calc, name="calculator", side_effect=SideEffect.READ_ONLY))

    async def json_pretty(text: str) -> str:
        try:
            return json.dumps(json.loads(text), indent=2, sort_keys=True)
        except Exception as exc:
            raise ToolError(f"invalid json: {exc}") from exc

    json_pretty.__doc__ = "Pretty-print a JSON string."

    async def py_ast_dump(source: str) -> str:
        try:
            tree = ast.parse(source)
            return ast.dump(tree, indent=2)[:16_384]
        except SyntaxError as exc:
            raise ToolError(f"syntax error: {exc}") from exc

    py_ast_dump.__doc__ = "Parse Python source and return an AST dump (no execution)."
    tools.extend(
        [
            Tool(json_pretty, name="json_pretty", side_effect=SideEffect.READ_ONLY),
            Tool(py_ast_dump, name="py_ast_dump", side_effect=SideEffect.READ_ONLY),
        ]
    )
    return tools


def data_toolkit() -> list[Tool]:
    """Lightweight numeric / JSON data tools."""

    async def stats_summary(values: list[float]) -> str:
        if not values:
            raise ToolError("values must be non-empty")
        return json.dumps(
            {
                "n": len(values),
                "mean": statistics.fmean(values),
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "sum": math.fsum(values),
            }
        )

    stats_summary.__doc__ = "Compute mean/stdev/min/max/sum for a list of numbers."

    async def json_path(text: str, key: str) -> str:
        data = json.loads(text)
        cur: Any = data
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise ToolError(f"key not found: {key}")
        return json.dumps(cur)

    json_path.__doc__ = "Fetch a dotted key path from a JSON object string."
    return [
        Tool(stats_summary, name="stats_summary", side_effect=SideEffect.READ_ONLY),
        Tool(json_path, name="json_path", side_effect=SideEffect.READ_ONLY),
    ]


def filesystem_toolkit(*, root: str | Path | None = None) -> list[Tool]:
    """Read-only filesystem tools sandboxed under ``root`` (cwd by default)."""
    base = Path(root or Path.cwd()).resolve()

    def _resolve(path: str) -> Path:
        target = (base / path).resolve()
        try:
            if not target.is_relative_to(base):
                raise ToolError("path escapes sandbox root")
        except AttributeError:
            # Python < 3.9 fallback
            if base != target and base not in target.parents:
                raise ToolError("path escapes sandbox root") from None
        return target

    async def list_dir(path: str = ".") -> str:
        target = _resolve(path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries[:500])

    list_dir.__doc__ = "List files under a sandboxed directory."

    async def read_text(path: str, max_chars: int = 32_768) -> str:
        target = _resolve(path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        return target.read_text(errors="replace")[:max_chars]

    read_text.__doc__ = "Read a text file inside the sandbox."
    return [
        Tool(list_dir, name="list_dir", side_effect=SideEffect.READ_ONLY),
        Tool(read_text, name="read_text", side_effect=SideEffect.READ_ONLY),
    ]


_TOOLKITS = {
    "web": web_toolkit,
    "code": code_toolkit,
    "data": data_toolkit,
    "filesystem": filesystem_toolkit,
}


def toolkit(name: str, **options: Any) -> list[Tool]:
    if name not in _TOOLKITS:
        raise ConfigurationError(
            f"unknown toolkit {name!r}",
            code="agents.toolkit_unknown",
            context={"available": sorted(_TOOLKITS)},
        )
    return _TOOLKITS[name](**options)


def catalog() -> dict[str, Any]:
    return {
        "kind": "agent_toolkits",
        "toolkits": sorted(_TOOLKITS),
        "factory": "aire.agents.toolkits.toolkit",
    }
