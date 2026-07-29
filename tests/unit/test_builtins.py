"""Builtin tool catalog and calculator."""

from __future__ import annotations

from aire.models.base import run_sync
from aire.tools.builtins import builtin_tools


def test_builtin_names_include_web_search_and_http_post() -> None:
    names = {t.name for t in builtin_tools()}
    assert "web_search" in names
    assert "http_post" in names
    assert "calculator" in names
    assert "http_get" in names
    assert "read_file" in names


def test_calculator_evaluates_arithmetic() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    result = run_sync(calc.execute({"expression": "2 + 3 * 4"}))
    assert result.ok
    assert float(result.output) == 14.0


def test_calculator_rejects_non_arithmetic() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    result = run_sync(calc.execute({"expression": "__import__('os').system('echo x')"}))
    assert not result.ok
