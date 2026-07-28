"""Registry, plugins, events, context, lifecycle, serialization."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aire.core.errors import DataError, NotFoundError, PluginError
from aire.core.events import EventBus
from aire.core.lifecycle import ResourceManager
from aire.core.plugins import PluginInfo, PluginManager
from aire.core.registry import Registry
from aire.core.serialization import iter_jsonl, read_yaml_file, write_jsonl
from aire.core.types import Ref


def test_registry_register_resolve() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register("alpha", lambda: "A")
    assert reg.create("alpha") == "A"
    assert reg.has("alpha")
    assert reg.names() == ["alpha"]


def test_registry_duplicate_rejected() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register("x", lambda: "1")
    with pytest.raises(PluginError):
        reg.register("x", lambda: "2")
    reg.register("x", lambda: "2", replace=True)
    assert reg.create("x") == "2"


def test_registry_missing_lists_available() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register("only", lambda: "1")
    with pytest.raises(NotFoundError) as excinfo:
        reg.create("nope")
    assert excinfo.value.context["available"] == ["only"]


def test_registry_decorator_form() -> None:
    reg: Registry[str] = Registry("thing")

    @reg.register("decorated")
    def _make() -> str:
        return "D"

    assert reg.create("decorated") == "D"


def test_ref_parsing() -> None:
    ref = Ref.parse("openai:gpt-4o-mini")
    assert ref.provider == "openai"
    assert ref.name == "gpt-4o-mini"
    assert str(ref) == "openai:gpt-4o-mini"
    with pytest.raises(Exception) as excinfo:
        Ref.parse("no-colon")
    assert getattr(excinfo.value, "code", "") == "ref.invalid"


def test_plugin_manager_contract(runtime: object) -> None:
    from aire.core.runtime import Runtime

    assert isinstance(runtime, Runtime)
    manager = PluginManager()

    class GoodPlugin:
        __name__ = "good"

        @staticmethod
        def register(rt: Runtime) -> PluginInfo:
            return PluginInfo(name="good", version="1.0", provides=["x"])

    info = manager.register_module(GoodPlugin(), runtime)
    assert info.name == "good"
    assert manager.loaded["good"].version == "1.0"

    class BadPlugin:
        __name__ = "bad"

        @staticmethod
        def register(rt: Runtime) -> PluginInfo:
            raise ValueError("broken")

    with pytest.raises(PluginError) as excinfo:
        manager.register_module(BadPlugin(), runtime)
    assert excinfo.value.code == "plugin.register_failed"


def test_event_bus_sync_and_async() -> None:
    bus = EventBus()
    seen: list[str] = []

    bus.subscribe("model.*", lambda e: seen.append(e.topic))
    bus.subscribe("model.call", lambda e: seen.append("exact"))

    bus.emit("model.call", {"x": 1})
    bus.emit("other.topic")
    assert seen == ["model.call", "exact"]
    assert len(bus.history) == 2

    async def _async() -> list[str]:
        got: list[str] = []

        async def handler(e: object) -> None:
            got.append("async")

        bus.subscribe("async.topic", handler)
        await bus.emit_async("async.topic")
        return got

    assert asyncio.run(_async()) == ["async"]


def test_resource_manager_lifo() -> None:
    order: list[str] = []

    async def _main() -> None:
        manager = ResourceManager()
        manager.track("first", lambda: order.append("first"))
        manager.track("second", lambda: order.append("second"))

        class Client:
            async def aclose(self) -> None:
                order.append("client")

        manager.track_resource("client", Client())
        assert manager.open_count == 3
        await manager.aclose()

    asyncio.run(_main())
    assert order == ["client", "second", "first"]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl", [{"a": 1}, {"a": 2}])
    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_yaml_safe_load_blocks_tags(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("x: 1\n")
    assert read_yaml_file(path) == {"x": 1}
    bad = tmp_path / "evil2.yaml"
    bad.write_text("!!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(DataError):
        read_yaml_file(bad)
