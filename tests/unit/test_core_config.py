"""Configuration loading, layering and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.config import Settings, _deep_merge
from aire.core.errors import ConfigurationError


def test_defaults() -> None:
    settings = Settings()
    assert settings.model.ref == "mock:echo"
    assert settings.model.embedder == "local:hashing"
    assert settings.safety.require_approval["high_impact"] is True


def test_load_yaml(tmp_path: Path) -> None:
    config = tmp_path / "aire.yaml"
    config.write_text(
        "project: demo\nmodel:\n  ref: openai:gpt-4o-mini\n  temperature: 0.2\n"
        "agent:\n  max_steps: 5\n"
    )
    settings = Settings.load(config)
    assert settings.project == "demo"
    assert settings.model.ref == "openai:gpt-4o-mini"
    assert settings.model.temperature == 0.2
    assert settings.agent.max_steps == 5


def test_overrides_beat_file(tmp_path: Path) -> None:
    config = tmp_path / "aire.yaml"
    config.write_text("project: file-name\nmodel:\n  ref: mock:echo\n")
    settings = Settings.load(config, overrides={"project": "override-name"})
    assert settings.project == "override-name"


def test_env_layering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "aire.yaml"
    config.write_text("project: base\n")
    monkeypatch.setenv("AIRE_PROJECT", "from-env")
    monkeypatch.setenv("AIRE_AGENT__MAX_STEPS", "9")
    settings = Settings.load(config)
    assert settings.project == "from-env"
    assert settings.agent.max_steps == 9


def test_invalid_config_raises_structured_error(tmp_path: Path) -> None:
    config = tmp_path / "aire.yaml"
    config.write_text("project: [unclosed\n")
    with pytest.raises(ConfigurationError) as excinfo:
        Settings.load(config)
    assert excinfo.value.code == "config.load_failed"


def test_deep_merge_nested() -> None:
    merged = _deep_merge(
        {"a": {"b": 1, "c": 2}, "d": 3},
        {"a": {"b": 10}, "e": 4},
    )
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}


def test_save_and_reload(tmp_path: Path) -> None:
    settings = Settings(project="roundtrip")
    path = settings.save(tmp_path / "aire.yaml")
    assert Settings.load(path).project == "roundtrip"


def test_project_file_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (tmp_path / "aire.yaml").write_text("project: discovered\n")
    monkeypatch.chdir(nested)
    settings = Settings.load(env=False)
    assert settings.project == "discovered"
