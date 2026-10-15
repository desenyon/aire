import json
import sys
import types

import pytest
from typer.testing import CliRunner

from aire.cli.main import app
from aire.observability.analytics import create_analytics


@pytest.mark.parametrize("prometheus", [False, True])
def test_invalid_analytics_source_fails(prometheus, monkeypatch):
    monkeypatch.setenv("AIRE_ANALYTICS", "missing_aire_analytics:source")
    result = CliRunner().invoke(app, ["analytics", *(["--prometheus"] if prometheus else [])])
    assert result.exit_code != 0
    assert "demo_mode" not in result.stdout


def test_real_analytics_source_is_loaded(monkeypatch):
    module = types.ModuleType("qa_analytics")
    module.source = create_analytics()
    module.source.record_run(model="qa:model", cost_usd=1.0, latency_ms=10.0, operation="qa")
    monkeypatch.setitem(sys.modules, "qa_analytics", module)
    monkeypatch.setenv("AIRE_ANALYTICS", "qa_analytics:source")
    result = CliRunner().invoke(app, ["analytics"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["demo_mode"] is False
    assert data["costs"]["by_model"] == {"qa:model": 1.0}


def test_no_analytics_source_stays_demo(monkeypatch):
    monkeypatch.delenv("AIRE_ANALYTICS", raising=False)
    result = CliRunner().invoke(app, ["analytics"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["demo_mode"] is True
