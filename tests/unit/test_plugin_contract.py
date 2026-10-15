from types import SimpleNamespace

import pytest

from aire.core.config import Settings
from aire.core.errors import ConfigurationError, PluginError
from aire.core.plugins import PluginManager
from aire.core.registry import Registry
from aire.core.runtime import Runtime


def test_plugin_import_failure_is_normalized(monkeypatch):
    from aire.core import plugins

    def broken():
        raise RuntimeError("bad plugin import")

    monkeypatch.setattr(
        plugins.metadata, "entry_points", lambda **kw: [SimpleNamespace(name="broken", load=broken)]
    )
    with pytest.raises(PluginError) as error:
        PluginManager().load_entry_points(Runtime(Settings()))
    assert error.value.code == "plugin.import_failed"


def test_registry_collision_is_not_silent():
    registry = Registry("qa")
    registry.register("one", lambda: 1)
    with pytest.raises(PluginError):
        registry.register("one", lambda: 2)
    assert registry.create("one") == 1


def test_skill_import_failure_is_not_silently_omitted(tmp_path):
    from aire.agents.skills import _load_skill_tools

    (tmp_path / "tools.py").write_text("raise RuntimeError('broken')")
    with pytest.raises(ConfigurationError, match="skill"):
        _load_skill_tools(tmp_path, {})


def test_unsupported_schema_fails_explicitly():
    from aire.tools.schema import schema_model

    with pytest.raises(ConfigurationError, match="unsupported"):
        schema_model(
            "qa", {"type": "object", "properties": {"x": {"allOf": [{"type": "integer"}]}}}
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"allOf": [{"type": "object"}]},
        {
            "type": "object",
            "properties": {"x": {"oneOf": [{"type": "number"}, {"type": "integer"}]}},
        },
    ],
)
def test_unsupported_composition_never_silently_accepts_invalid_arguments(schema):
    from aire.tools.schema import schema_model

    with pytest.raises(ConfigurationError, match="unsupported"):
        schema_model("qa", schema)
