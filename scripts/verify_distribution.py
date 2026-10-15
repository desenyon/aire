"""Install a built wheel outside the checkout and verify the distribution contract.

Usage: python scripts/verify_distribution.py dist/aire_ai-*.whl
Requires uv on PATH; uses a disposable virtual environment and no provider services.
"""

# ruff: noqa: S603
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

SMOKE = """
import asyncio
import importlib.metadata as m
import sys
from pathlib import Path
from aire import AI, __version__
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.models.base import run_sync
assert m.version("aire-ai") == __version__
assert Path(__import__("aire").__file__).with_name("py.typed").is_file()
assert run_sync(AI.models.use_sync("mock:echo").ask("wheel contract")) == "wheel contract"
assert AI.describe()["library"] == "aire"
runtime = Runtime(Settings(), discover_plugins=True)
assert {"mock", "openai", "anthropic"} <= set(runtime.plugins.describe())
asyncio.run(runtime.aclose())
optional = {"torch", "tensorflow", "keras", "sklearn", "pandas", "fastapi", "numpy"}
assert not set(sys.modules) & optional
print("wheel contract passed", __version__)
"""


def main() -> None:
    wheel = pathlib.Path(sys.argv[1]).resolve()
    uv = shutil.which("uv")
    if not uv or not wheel.is_file():
        raise SystemExit("uv and a built wheel path are required")
    with tempfile.TemporaryDirectory(prefix="aire-wheel-") as directory:
        environment = pathlib.Path(directory) / "environment"
        subprocess.run([uv, "venv", str(environment), "--python", sys.executable], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([uv, "pip", "install", "--python", str(python), str(wheel)], check=True)
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=directory, check=True)
        cli = python.with_name("aire.exe" if sys.platform == "win32" else "aire")
        subprocess.run([str(cli), "doctor"], cwd=directory, check=True, capture_output=True)
        subprocess.run([uv, "pip", "check", "--python", str(python)], check=True)


if __name__ == "__main__":
    main()
