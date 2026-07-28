"""CLI commands: init, run, inspect, doctor, evaluate."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aire.cli.main import app

runner = CliRunner()


def test_init_scaffolds_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "myproj", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    root = tmp_path / "myproj"
    assert (root / "aire.yaml").is_file()
    assert (root / "app.py").is_file()
    assert (root / "docs").is_dir()
    assert (root / "evals" / "questions.jsonl").is_file()


def test_init_refuses_nonempty(tmp_path: Path) -> None:
    root = tmp_path / "taken"
    root.mkdir()
    (root / "file.txt").write_text("x")
    result = runner.invoke(app, ["init", "taken", "--dir", str(tmp_path)])
    assert result.exit_code == 1


def test_run_with_mock_model() -> None:
    result = runner.invoke(app, ["run", "hello aire", "--model", "mock:echo"])
    assert result.exit_code == 0, result.output
    assert "hello aire" in result.output


def test_inspect() -> None:
    result = runner.invoke(app, ["inspect", "models"])
    assert result.exit_code == 0
    assert "mock" in result.output


def test_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert '"healthy": true' in result.output


def test_evaluate_command(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"input": "x", "expected": "x"}\n')
    result = runner.invoke(app, ["evaluate", str(dataset), "--model", "mock:echo"])
    assert result.exit_code == 0, result.output
    assert "exact_match" in result.output or "accuracy" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()
