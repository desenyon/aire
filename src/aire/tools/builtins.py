"""Builtin tools available in every project.

Security notes:
- ``calculator`` evaluates arithmetic via a restricted AST walker — never eval().
- ``read_file`` and ``list_files`` are confined to an explicit sandbox root.
- ``http_get`` / ``http_post`` / ``web_search`` perform outbound network access
  and are therefore classified ``external_side_effect`` so policies can gate them.
- ``web_search`` scrapes DuckDuckGo HTML lite — not an official search API.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from aire.core.errors import SafetyError, ToolError
from aire.tools.tool import Tool
from aire.tools.types import SideEffect

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
            raise ToolError("exponent too large")
        result = _BIN_OPS[type(node.op)](left, right)
        if abs(result) > 1e15:
            raise ToolError("result magnitude too large")
        return float(result)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _calculator(expression: str) -> float:
    """Evaluate an arithmetic expression (+, -, *, /, //, %, **, parentheses)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc}", cause=exc) from exc
    return _eval_node(tree.body)


def _confine(path: str, sandbox_root: str) -> Path:
    root = Path(sandbox_root).resolve()
    resolved = Path(path).resolve()
    if root != resolved and root not in resolved.parents:
        raise SafetyError(
            f"path {resolved} escapes sandbox root {root}",
            code="safety.path_traversal",
            context={"path": str(resolved), "root": str(root)},
        )
    return resolved


def _read_file(path: str, sandbox_root: str = ".") -> str:
    """Read a UTF-8 text file inside the sandbox root."""
    resolved = _confine(path, sandbox_root)
    if not resolved.is_file():
        raise ToolError(f"not a file: {resolved}")
    if resolved.stat().st_size > 1_000_000:
        raise ToolError("file too large (>1MB)")
    return resolved.read_text(errors="replace")


def _list_files(
    directory: str = ".",
    pattern: str = "**/*",
    sandbox_root: str = ".",
) -> list[str]:
    """List files under a directory confined to the sandbox root (relative paths)."""
    root = Path(sandbox_root).resolve()
    target = _confine(directory, sandbox_root)
    if not target.is_dir():
        raise ToolError(f"not a directory: {target}")
    return sorted(
        str(p.relative_to(root)) for p in target.glob(pattern) if p.is_file()
    )[:500]


async def _http_get(url: str) -> str:
    """Fetch a URL over HTTP GET and return the response body (truncated to 64KB)."""
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text[:65_536]


async def _http_post(
    url: str,
    body: str = "",
    content_type: str = "application/json",
) -> str:
    """POST ``body`` to ``url`` and return the response body (truncated to 64KB)."""
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        response = await client.post(
            url,
            content=body.encode("utf-8") if isinstance(body, str) else body,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()
        return response.text[:65_536]


_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


async def _web_search(query: str, max_results: int = 5) -> str:
    """Search via DuckDuckGo HTML lite scrape (not an official API).

    Fetches ``https://html.duckduckgo.com/html/?q=...``, parses result titles
    and URLs with a simple regex, and returns a JSON string of
    ``[{"title": ..., "url": ...}, ...]``. Raises :class:`ToolError` if the
    network request fails.
    """
    import httpx

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception as exc:
        raise ToolError(f"web_search network failure: {exc}", cause=exc) from exc

    results: list[dict[str, str]] = []
    for match in _RESULT_RE.finditer(html):
        href = match.group(1).strip()
        title = _TAG_RE.sub("", match.group(2)).strip()
        if href and title:
            results.append({"title": title, "url": href})
        if len(results) >= max(1, int(max_results)):
            break
    return json.dumps(results)


def _current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def builtin_tools() -> list[Tool]:
    """Instantiate the standard tool set."""
    return [
        Tool(_calculator, name="calculator", side_effect=SideEffect.READ_ONLY),
        Tool(_read_file, name="read_file", side_effect=SideEffect.READ_ONLY),
        Tool(_list_files, name="list_files", side_effect=SideEffect.READ_ONLY),
        Tool(_http_get, name="http_get", side_effect=SideEffect.EXTERNAL_SIDE_EFFECT),
        Tool(_http_post, name="http_post", side_effect=SideEffect.EXTERNAL_SIDE_EFFECT),
        Tool(
            _web_search,
            name="web_search",
            side_effect=SideEffect.EXTERNAL_SIDE_EFFECT,
            description=(
                "Search the web via DuckDuckGo HTML scrape (not an official API). "
                "Returns JSON list of {title, url}."
            ),
        ),
        Tool(_current_time, name="current_time", side_effect=SideEffect.READ_ONLY),
    ]
