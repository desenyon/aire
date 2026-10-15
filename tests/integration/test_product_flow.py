import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aire.cli.main import app
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.evaluation.gates import check_gates
from aire.knowledge_assistant import Assistant
from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.project.lock import create_lock, load_lock, write_lock


def test_builder_index_ask_evaluate_gate_and_deploy_share_configuration(tmp_path):
    runtime = Runtime(Settings())
    runtime.model_providers.register("qa", lambda name, **kw: EchoModel(prefix="configured: "))
    assistant = Assistant("flow", runtime).documents(
        [{"text": "Aire access uses short lived tokens."}]
    )
    assistant.knowledge()
    assistant.model("qa:answer")
    assert assistant.index().chunks > 0
    assert assistant.ask("How does access work?").citations
    report = assistant.evaluate(
        [{"input": "How does access work?", "expected": "tokens"}], metrics=["contains"]
    )
    assert check_gates(report, [{"metric": "contains", "min": 1}]).passed
    assert not check_gates(report, [{"metric": "contains", "min": 2}]).passed
    with TestClient(assistant.deploy()) as client:
        response = client.post("/v1/ask", json={"question": "How does access work?"})
        assert response.json()["answer"].startswith("configured: ")
        assert response.json()["citations"]
    run_sync(runtime.aclose())


def test_cli_project_and_reference_lock_roundtrip(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["init", "qa", "--dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["init", "qa", "--dir", str(tmp_path)]).exit_code != 0
    assert runner.invoke(app, ["run", "offline hello", "--model", "mock:echo"]).exit_code == 0
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    lock = create_lock("qa", model="mock:echo", embedder="local:hashing")
    path = write_lock(lock, tmp_path / "aire.lock")
    restored = load_lock(path)
    assert restored.get("model") == "mock:echo"
    assert restored.refs("embedder") == ["local:hashing"]


def test_cli_scaffold_serves_indexed_knowledge(tmp_path, monkeypatch):
    import uvicorn

    runner = CliRunner()
    runner.invoke(app, ["init", "server", "--dir", str(tmp_path)])
    monkeypatch.chdir(tmp_path / "server")
    responses = []

    def run(api, **kwargs):
        with TestClient(api) as client:
            responses.append(client.post("/v1/ask", json={"question": "What is server?"}))

    monkeypatch.setattr(uvicorn, "run", run)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.exception
    assert responses[0].status_code == 200
    assert responses[0].json()["citations"]


def test_graphrag_ingest_query_and_communities():
    from aire.graph.extract import LexicalGraphExtractor
    from aire.graph.pipeline import KnowledgeGraph
    from aire.models.builtin import HashingEmbedder
    from aire.rag.types import Document

    async def check():
        async with Runtime(Settings()) as runtime:
            graph = KnowledgeGraph(
                runtime,
                model=EchoModel(),
                extractor=LexicalGraphExtractor(),
                embedder=HashingEmbedder(),
            )
            try:
                report = await graph.ingest(
                    [
                        Document(
                            id="graph-doc", text="Alice works with Acme. Bob also works with Acme."
                        )
                    ]
                )
                assert report.entities >= 3 and report.relations >= 2
                answer = await graph.query("Alice")
                assert answer.citations
                assert answer.metadata["communities"]["communities"]
                assert "Acme" in answer.text
            finally:
                await graph.store.aclose()

    run_sync(check())


def test_cli_evaluation_manifest_and_scale_artifacts(tmp_path):
    import json

    runner = CliRunner()
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"input":"hello","expected":"hello"}\n')
    output = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(cases),
            "--model",
            "mock:echo",
            "--metrics",
            "exact_match",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(output.read_text())["results"][0]["metrics"][0]["score"] == 1
    inspected = runner.invoke(app, ["inspect", "all"])
    assert inspected.exit_code == 0
    assert {"models", "tools", "plugins", "config"} <= json.loads(inspected.stdout).keys()
    target = tmp_path / "deployment"
    deployed = runner.invoke(app, ["deploy", "scale", str(target)])
    assert deployed.exit_code == 0, deployed.exception
    assert (target / "requirements.in").is_file()
    assert runner.invoke(app, ["evaluate", str(tmp_path / "absent.jsonl")]).exit_code != 0


def test_scaffold_preserves_existing_files(tmp_path):
    runner = CliRunner()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "main.py").write_text("# my program\n")
    result = runner.invoke(app, ["scaffold", "agent", "--name", "existing", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert (target / "main.py").read_text() == "# my program\n"


@pytest.mark.parametrize("kind", ["rag", "agent", "finetune", "gateway", "workflow"])
def test_each_scaffold_recipe_executes_its_named_outcome(tmp_path, kind):
    import runpy

    runner = CliRunner()
    created = runner.invoke(app, ["scaffold", kind, "--name", kind, "--dir", str(tmp_path)])
    assert created.exit_code == 0, created.exception
    values = runpy.run_path(str(tmp_path / kind / "main.py"))
    if kind == "rag":
        assert values["answer"].citations
    elif kind == "finetune":
        assert values["result"].epochs_completed > 0
    elif kind == "workflow":
        assert values["result"].ok
    elif kind == "agent":
        assert values["agent"].state.messages
    else:
        with TestClient(values["app"]) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock:echo",
                    "messages": [{"role": "user", "content": "recipe request"}],
                },
            )
            assert response.status_code == 200


def test_ui_uses_supplied_runtime_for_all_operational_data():
    from aire.ai import _ObserveNamespace
    from aire.ui import create_ui_app

    runtime = Runtime(Settings(project="ui-tenant"))
    observe = _ObserveNamespace(runtime)
    observe.metrics.record_cost(2.5, model="qa:model")
    with observe.tracer().span("tenant-only-trace"):
        pass
    runtime.events.emit("tenant.event", data={"value": "owned"})
    with TestClient(create_ui_app(runtime=runtime)) as client:
        assert client.get("/api/traces").json()["traces"][0]["name"] == "tenant-only-trace"
        assert client.get("/api/costs").json()["costs"]["total_usd"] == 2.5
        assert client.get("/api/metrics").json()["metrics"]["counters"]
        assert client.get("/api/events").json()["events"]
        manifest = client.get("/api/manifest").json()
        assert manifest["runtime"]["project"] == "ui-tenant"
        assert manifest["ai"]["runtime"]["project"] == "ui-tenant"
    assert _ObserveNamespace(runtime).metrics is observe.metrics
    run_sync(runtime.aclose())
