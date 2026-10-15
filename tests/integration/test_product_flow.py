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
