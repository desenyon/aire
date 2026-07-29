"""Sandbox path confinement for read_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.errors import SafetyError
from aire.models.base import run_sync
from aire.tools.builtins import builtin_tools


def test_read_file_escape_raises(sandbox_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    reader = next(t for t in builtin_tools() if t.name == "read_file")
    # Escape attempt relative to sandbox
    result = run_sync(
        reader.execute({"path": str(outside), "sandbox_root": str(sandbox_root)})
    )
    assert not result.ok
    # Prefer SafetyError in cause/message; Tool wraps some failures
    err = (result.error or "").lower()
    assert "sandbox" in err or "escape" in err or "path" in err


def test_read_file_inside_sandbox_ok(sandbox_root: Path) -> None:
    reader = next(t for t in builtin_tools() if t.name == "read_file")
    result = run_sync(
        reader.execute(
            {"path": str(sandbox_root / "hello.txt"), "sandbox_root": str(sandbox_root)}
        )
    )
    assert result.ok
    assert "hello from sandbox" in str(result.output)


def test_confine_raises_safety_error(sandbox_root: Path, tmp_path: Path) -> None:
    from aire.tools import builtins as builtins_mod

    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(SafetyError):
        builtins_mod._confine(str(outside), str(sandbox_root))
