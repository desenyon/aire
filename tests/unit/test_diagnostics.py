import json

import httpx
import pytest
from typer.testing import CliRunner

from aire import __version__
from aire.cli.main import app


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "aire.yaml"
    path.write_text("project: diagnostics-test\n")
    return path


def test_doctor_smoke_is_offline_and_persists_real_outcomes(isolated_config, monkeypatch, tmp_path):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("offline doctor attempted network")

    monkeypatch.setattr(httpx.AsyncClient, "send", network_forbidden)
    monkeypatch.setattr(httpx.Client, "send", network_forbidden)
    path = tmp_path / "reports" / "doctor.json"
    result = CliRunner().invoke(app, ["doctor", "--smoke", "--output", str(path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert json.loads(path.read_text()) == report
    assert report["schema_version"] == 1
    assert report["version"] == __version__
    assert report["healthy"] and not report["live"]
    checks = {item["check"]: item for item in report["checks"]}
    for name in ("smoke:generation", "smoke:embedding", "smoke:tool", "smoke:rag"):
        assert checks[name]["status"] == "passed"
    assert checks["smoke:tool"]["detail"] == "2 + 3 = 5"
    assert checks["smoke:rag"]["detail"] == "retrieved=1; source=doctor-fixture"
    assert checks["credentials:openai"]["status"] == "skipped"
    assert report["skipped"] >= 3


def test_doctor_invalid_config_reports_failure_without_secret_values(isolated_config, tmp_path):
    isolated_config.write_text("model:\n  max_tokens: secret-invalid-value\n")
    path = tmp_path / "failed.json"
    result = CliRunner().invoke(app, ["doctor", "--output", str(path)])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["healthy"]
    assert "secret-invalid-value" not in result.output
    assert json.loads(path.read_text()) == report
    check = next(item for item in report["checks"] if item["check"] == "config:loaded")
    assert check["status"] == "failed"
    assert "config.validation" in check["detail"]


def test_doctor_smoke_failure_is_not_healthy(isolated_config, monkeypatch):
    from aire.models.builtin import EchoModel

    async def broken(self, request):
        raise RuntimeError("sensitive provider response")

    monkeypatch.setattr(EchoModel, "generate", broken)
    result = CliRunner().invoke(app, ["doctor", "--smoke"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["healthy"]
    assert any(
        item["check"] == "smoke:generation" and item["status"] == "failed"
        for item in report["checks"]
    )
    assert "sensitive provider response" not in result.stdout


def test_live_doctor_uses_config_credentials_and_closes_clients(isolated_config, monkeypatch):
    isolated_config.write_text("providers:\n  openai:\n    api_key: configured-secret\n")
    seen = []
    closed = []
    original_close = httpx.AsyncClient.aclose

    async def send(client, request, **kwargs):
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"data": []}, request=request)

    async def close(client):
        closed.append(True)
        await original_close(client)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            200, json={"models": []}, request=httpx.Request("GET", "http://localhost/api/tags")
        ),
    )
    result = CliRunner().invoke(app, ["doctor", "--live"])
    assert result.exit_code == 0, result.output
    assert ("/v1/models", "Bearer configured-secret") in seen
    assert closed
    report = json.loads(result.stdout)
    check = next(item for item in report["checks"] if item["check"].startswith("live:openai:"))
    assert check["status"] == "passed"
    assert "configured-secret" not in result.stdout
