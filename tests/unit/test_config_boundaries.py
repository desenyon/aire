import json

import pytest
from typer.testing import CliRunner

from aire import AI
from aire.cli.main import app
from aire.core.config import Settings
from aire.core.errors import ConfigurationError


@pytest.mark.parametrize("directory", [False, True])
def test_explicit_missing_config_never_uses_defaults(tmp_path, monkeypatch, directory):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "aire.yaml").write_text("project: unrelated\n")
    requested = tmp_path / ("empty" if directory else "missing.yaml")
    if directory:
        requested.mkdir()
    with pytest.raises(ConfigurationError) as error:
        Settings.load(requested, env=False)
    assert error.value.code == "config.not_found"
    assert str(requested) in error.value.context["path"]


def test_implicit_discovery_and_explicit_directory_still_load(tmp_path, monkeypatch):
    (tmp_path / "aire.yaml").write_text("project: chosen\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    assert Settings.load(env=False).project == "chosen"
    assert Settings.load(tmp_path, env=False).project == "chosen"


def test_inspection_masks_nested_secrets_and_custom_fields():
    settings = Settings(
        gateway={"auth_token": "plain-gateway-secret"},
        model={"options": {"Authorization": "plain-header-secret", "api-key": "plain-api-secret"}},
        observability={"mask_fields": ["private_value"]},
        custom={"items": [{"private_value": "custom-secret", "safe": "visible"}]},
        providers={"openai": {"api_key": "provider-secret"}},
    )
    AI.configure(settings)
    result = CliRunner().invoke(app, ["inspect", "config"])
    assert result.exit_code == 0
    for secret in (
        "plain-gateway-secret",
        "plain-header-secret",
        "plain-api-secret",
        "custom-secret",
        "provider-secret",
    ):
        assert secret not in result.stdout
    report = json.loads(result.stdout)["config"]
    assert report["custom"]["items"][0]["safe"] == "visible"
    assert report["gateway"]["auth_token"] == "[REDACTED]"
    assert settings.gateway.auth_token == "plain-gateway-secret"


def test_unknown_inspection_selector_fails():
    result = CliRunner().invoke(app, ["inspect", "typo"])
    assert result.exit_code == 2
    assert "typo" in result.output
