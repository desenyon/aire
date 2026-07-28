"""0.2.4 hardening tests: workflow resume, approval policies, new loaders."""

from __future__ import annotations

import pytest

from aire.agents.approvals import InteractiveApprover, RuleApprover
from aire.data.loaders import html_to_text, load
from aire.models.types import ToolCall
from aire.tools.types import SideEffect, ToolSpec
from aire.workflows.graph import Workflow
from aire.workflows.types import NodeStatus
from tests.conftest import arun

# -- workflow checkpoint resume -------------------------------------------------------


def test_workflow_resume_continues_pending_nodes(tmp_path) -> None:
    calls: list[str] = []
    fail_b = {"value": True}

    def node_a(input, ctx):
        calls.append("a")
        return 1

    def node_b(input, ctx):
        calls.append("b")
        if fail_b["value"]:
            raise RuntimeError("simulated crash")
        return input + 1

    def node_c(input, ctx):
        calls.append("c")
        return input + 1

    def build() -> Workflow:
        graph = Workflow("pipe", checkpoint_path=tmp_path / "wf.json")
        graph.add("a", node_a).add("b", node_b).add("c", node_c)
        graph.connect("a", "b").connect("b", "c")
        graph.entry("a")
        return graph

    first = arun(build().run())
    assert not first.ok
    assert calls == ["a", "b"]
    assert first.outputs.get("a") == 1  # a's output survived in the checkpoint

    fail_b["value"] = False
    resumed = arun(build().resume())
    assert resumed.ok
    assert calls == ["a", "b", "b", "c"]  # a never re-ran; b retried; c ran
    assert resumed.output == 3

    statuses = {rec.name: rec.status for rec in resumed.records}
    assert statuses == {
        "a": NodeStatus.COMPLETED,
        "b": NodeStatus.COMPLETED,
        "c": NodeStatus.COMPLETED,
    }


def test_workflow_resume_respects_untaken_branches(tmp_path) -> None:
    def node_a(input, ctx):
        return "left"

    graph = Workflow("branches", checkpoint_path=tmp_path / "wf.json")
    graph.add("a", node_a)
    graph.add("left", lambda input, ctx: f"L:{input}")
    graph.add("right", lambda input, ctx: f"R:{input}")
    graph.connect("a", "left", when=lambda out: out == "left")
    graph.connect("a", "right", when=lambda out: out == "right")
    graph.entry("a")
    result = arun(graph.run())
    assert result.ok
    assert result.outputs["left"] == "L:left"
    statuses = {rec.name: rec.status for rec in result.records}
    assert statuses["right"] == NodeStatus.SKIPPED


def test_load_checkpoint_missing_raises(tmp_path) -> None:
    from aire.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        Workflow.load_checkpoint(tmp_path / "nope.json")


# -- approval policies -------------------------------------------------------------------


def _spec(side_effect: SideEffect, name: str = "tool") -> ToolSpec:
    return ToolSpec(name=name, side_effect=side_effect)


def _call(name: str = "tool") -> ToolCall:
    return ToolCall(id="c1", name=name, arguments={"x": 1})


def test_rule_approver_thresholds() -> None:
    approver = RuleApprover(auto_approve_below=SideEffect.REVERSIBLE_WRITE)
    assert approver(_call(), _spec(SideEffect.READ_ONLY)) is True
    assert approver(_call(), _spec(SideEffect.REVERSIBLE_WRITE)) is False
    assert approver(_call(), _spec(SideEffect.EXTERNAL_SIDE_EFFECT)) is False
    assert len(approver.decisions) == 3  # audit trail


def test_rule_approver_overrides_win() -> None:
    approver = RuleApprover(allow={"dangerous_but_trusted"}, deny={"read_secrets"})
    assert (
        approver(
            _call("dangerous_but_trusted"),
            _spec(SideEffect.EXTERNAL_SIDE_EFFECT, "dangerous_but_trusted"),
        )
        is True
    )
    assert approver(_call("read_secrets"), _spec(SideEffect.READ_ONLY, "read_secrets")) is False


def test_interactive_approver_session_memory(monkeypatch) -> None:
    answers = iter(["a"])  # "always" — remembered afterwards
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    approver = InteractiveApprover()
    spec = _spec(SideEffect.EXTERNAL_SIDE_EFFECT, "deploy")
    assert arun(approver(_call("deploy"), spec)) is True
    # second call: no prompt consumed, answered from session memory
    assert arun(approver(_call("deploy"), spec)) is True


def test_interactive_approver_deny(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    approver = InteractiveApprover()
    assert arun(approver(_call(), _spec(SideEffect.READ_ONLY))) is False


# -- loaders -----------------------------------------------------------------------------


def test_html_to_text_strips_markup() -> None:
    html = """
    <html><head><style>body { color: red }</style><title>Ignored</title></head>
    <body><h1>Refunds &amp; Returns</h1>
    <script>alert('x')</script>
    <p>Refunds take <b>5 days</b>.</p><ul><li>First</li><li>Second</li></ul>
    </body></html>
    """
    text = html_to_text(html)
    assert "Refunds & Returns" in text
    assert "5 days" in text
    assert "alert" not in text
    assert "color: red" not in text
    assert "<" not in text


def test_load_html_file(tmp_path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<html><body><p>Hello <b>world</b></p></body></html>")
    dataset = load(page)
    assert dataset.records[0].text == "Hello world"
    assert dataset.records[0].metadata["filename"] == "page.html"


def test_load_directory_includes_html(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("plain text")
    (tmp_path / "b.html").write_text("<p>markup text</p>")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    dataset = load(tmp_path)
    texts = {r.text for r in dataset}
    assert texts == {"plain text", "markup text"}


def test_parquet_requires_pandas(tmp_path) -> None:
    import importlib.util

    if importlib.util.find_spec("pandas") is not None:
        pytest.skip("pandas installed")
    from aire.core.errors import DataError

    target = tmp_path / "data.parquet"
    target.write_bytes(b"PAR1")  # content irrelevant: pandas check fires first
    with pytest.raises(DataError, match="pip install"):
        load(target)
