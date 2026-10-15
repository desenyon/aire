"""Run repeatable repository checks and retain source-bound JSON evidence.

Provision dependencies first. Reports are evidence for the selected profile,
not a certification of untested services, model quality or production scale.
"""

# ruff: noqa: S603, S607
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def source_state(root: Path) -> dict[str, Any]:
    """Fingerprint all tracked and nonignored source, including untracked inputs."""
    digest = hashlib.sha256()
    names = sorted(
        set(_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0"))
        - {b""}
    )
    for name in names:
        path = root / os.fsdecode(name)
        digest.update(name + b"\0")
        if path.is_symlink():
            digest.update(b"link\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(str(path.stat().st_mode & 0o777).encode())
        else:
            digest.update(b"missing\0")
        digest.update(b"\0")
    return {
        "commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=normal").strip()),
        "fingerprint": digest.hexdigest(),
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stop(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait()


class AuditInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int) -> None:
        super().__init__(f"audit interrupted by signal {signum}")
        self.signum = signum


@contextmanager
def _termination_signals() -> Iterator[None]:
    """Route catchable CLI termination through process-group cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: FrameType | None) -> NoReturn:
        raise AuditInterrupted(signum)

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in signals}
    try:
        for signum in signals:
            signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _execute(check: Check, root: Path, log: Path, timeout_seconds: float) -> tuple[str, int | None]:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        KERAS_BACKEND="torch",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with log.open("w", encoding="utf-8") as file:
        try:
            process = subprocess.Popen(
                check.command,
                cwd=root,
                env=env,
                stdout=file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            file.write(f"could not start check ({type(exc).__name__})\n")
            return "failed", None
        try:
            returncode = process.wait(timeout=timeout_seconds)
            return ("passed" if returncode == 0 else "failed"), returncode
        except subprocess.TimeoutExpired:
            _stop(process)
            return "timed_out", process.returncode
        except BaseException:
            _stop(process)
            raise


def _validate_output(root: Path, output: Path) -> None:
    if output.is_relative_to(root):
        relative = str(output.relative_to(root))
        log_relative = str(output.parent.relative_to(root) / (output.stem + "-logs"))
        for path in (relative, log_relative):
            ignored = (
                subprocess.run(
                    ["git", "-C", str(root), "check-ignore", "-q", path], check=False
                ).returncode
                == 0
            )
            if not ignored:
                raise ValueError(
                    "audit reports/logs must be outside the repo or in an ignored directory"
                )


def run_audit(
    root: Path, checks: list[Check], output: Path, *, timeout_seconds: float = 600
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    if not checks or len({check.name for check in checks}) != len(checks):
        raise ValueError("provide nonempty, uniquely named checks")
    if any(
        not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", check.name) or not check.command
        for check in checks
    ):
        raise ValueError("checks require safe names and nonempty command arguments")
    _validate_output(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem + "-logs")
    logs.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "source_start": source_state(root),
        "source_end": None,
        "source_changed": False,
        "completed": False,
        "healthy": False,
        "checks": [
            {
                "name": check.name,
                "command": check.command,
                "status": "not_run",
                "log": str(logs / (check.name + ".log")),
                "returncode": None,
            }
            for check in checks
        ],
    }
    _write_report(output, report)
    with _termination_signals():
        return _run_checks(root, checks, output, report, timeout_seconds)


def _run_checks(
    root: Path, checks: list[Check], output: Path, report: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    try:
        for check, entry in zip(checks, report["checks"], strict=True):
            entry["source_before"] = source_state(root)
            report["source_changed"] |= entry["source_before"] != report["source_start"]
            entry["status"] = "running"
            _write_report(output, report)
            started = time.monotonic()
            try:
                entry["status"], entry["returncode"] = _execute(
                    check, root, Path(entry["log"]), timeout_seconds
                )
            except BaseException:
                entry["status"] = "interrupted"
                raise
            finally:
                entry["duration_seconds"] = round(time.monotonic() - started, 3)
                entry["source_after"] = source_state(root)
                report["source_changed"] |= entry["source_after"] != report["source_start"]
                _write_report(output, report)
            print(f"{entry['name']}: {entry['status']} ({entry['duration_seconds']}s)", flush=True)
        report["completed"] = True
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["source_end"] = source_state(root)
        report["source_changed"] |= report["source_end"] != report["source_start"]
        report["healthy"] = (
            report["completed"]
            and not report["source_changed"]
            and all(entry["status"] == "passed" for entry in report["checks"])
        )
        _write_report(output, report)
    return report


def optional_environment() -> None:
    modules = (
        "sklearn",
        "skops",
        "torch",
        "keras",
        "xgboost",
        "lightgbm",
        "catboost",
        "peft",
        "transformers",
        "pypdf",
        "PIL",
        "pytesseract",
        "pypdfium2",
    )
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    missing += [binary for binary in ("ffmpeg", "tesseract") if shutil.which(binary) is None]
    if missing:
        raise SystemExit("optional audit prerequisites missing: " + ", ".join(missing))
    print("all optional CPU test prerequisites found")


def build_checks(profile: str) -> list[Check]:
    python = sys.executable
    if profile == "optional":
        return [
            Check(
                "optional-environment",
                [
                    python,
                    "-c",
                    "from scripts.audit import optional_environment; optional_environment()",
                ],
            ),
            Check("optional-tests", [python, "-m", "pytest", "-m", "not live"]),
        ]
    if profile != "core":
        raise ValueError("unknown audit profile")
    checks = [
        Check("lock", ["uv", "lock", "--check"]),
        Check("lint", [python, "-m", "ruff", "check", "src", "tests", "scripts"]),
        Check("format", [python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
        Check("types", [python, "-m", "mypy"]),
        Check("tests", [python, "-m", "pytest", "-m", "not live and not optional"]),
        Check("doctor", [python, "-m", "aire.cli.main", "doctor", "--smoke"]),
        Check("docs", [python, "-m", "mkdocs", "build", "--strict"]),
        Check("release-metadata", [python, "scripts/release.py", "check"]),
        Check("build", [python, "-m", "build", "--no-isolation"]),
        Check(
            "wheel",
            [
                python,
                "-c",
                "from pathlib import Path; import subprocess,sys; "
                "from aire import __version__; "
                'wheel=Path("dist")/f"aire_ai-{__version__}-py3-none-any.whl"; '
                'subprocess.run([sys.executable,"scripts/verify_distribution.py",str(wheel)],check=True)',
            ],
        ),
    ]
    checks.extend(
        Check("example-" + name.replace("_", "-"), [python, f"examples/{name}/main.py"])
        for name in ("reliable_local", "agent_tools", "gateway_offline", "eval_gates")
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "optional"), default="core")
    parser.add_argument("--output", type=Path, default=Path(".audit/report.json"))
    parser.add_argument("--timeout", type=float, default=600, help="Per-check timeout in seconds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        report = run_audit(
            root, build_checks(args.profile), args.output, timeout_seconds=args.timeout
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"audit could not complete: {exc}\n")
    except KeyboardInterrupt as exc:
        code = 128 + exc.signum if isinstance(exc, AuditInterrupted) else 130
        parser.exit(code, "audit interrupted; inspect the incomplete report\n")
    print(f"evidence: {args.output.resolve()}", flush=True)
    raise SystemExit(0 if report["healthy"] else 1)


if __name__ == "__main__":
    main()
