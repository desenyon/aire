"""The flagship vertical slice, fully offline:

documents → chunk → embed → store → retrieve → answer+citations → evaluate →
trace → deploy (FastAPI test client + artifacts).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.ai import AI
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.knowledge_assistant import Assistant
from aire.rag.store import register as register_local_store


@pytest.fixture()
def assistant(tmp_path: Path) -> Assistant:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "# Refunds\n\nCustomers may return products within 30 days for a full refund. "
        "After 30 days, store credit is offered instead."
    )
    (docs / "auth.md").write_text(
        "# Authentication\n\nThe API uses OAuth2 bearer tokens. Tokens expire after "
        "one hour and must be refreshed via the /token endpoint."
    )
    runtime = Runtime(Settings(project="slice"))
    register_local_store(runtime)
    return (
        Assistant("slice", runtime)
        .documents(docs)
        .model("mock:echo")
        .vector_store("local")
        .citations(True)
    )


def test_index_and_ask(assistant: Assistant) -> None:
    report = assistant.index()
    assert report.chunks >= 2
    answer = assistant.ask("How long do I have to return a product?")
    assert answer.citations
    assert any(
        "refund" in c.excerpt.lower() or "return" in c.excerpt.lower() for c in answer.citations
    )
    # echo model returns the grounded prompt — which must embed the context
    assert "30 days" in answer.text


def test_evaluate(assistant: Assistant) -> None:
    assistant.index()
    report = assistant.evaluate(
        [{"input": "What is the refund window?", "expected": "30 days"}],
        metrics=["contains", "groundedness"],
    )
    assert report.total == 1
    assert report.metric_summary()["contains"]["mean"] == 1.0


def test_tracing_captures_rag(assistant: Assistant) -> None:
    from aire.ai import _ObserveNamespace

    observe = _ObserveNamespace(assistant.runtime)
    tracer = observe.tracer()
    assistant.index()
    assistant.ask("How do tokens expire?")
    names = [r.name for r in tracer.records()]
    assert "rag.ask" in names


def test_deploy_fastapi(assistant: Assistant) -> None:
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    assistant.index()
    app = assistant.deploy()
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"
    manifest = client.get("/manifest").json()
    assert manifest["kind"] == "knowledge"
    response = client.post("/v1/ask", json={"question": "How do tokens expire?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body and body["citations"]


def test_deploy_auth_and_rate_limit(assistant: Assistant) -> None:
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    assistant.index()
    app = assistant.deploy(auth_token="s3cret", rate_limit_per_minute=2)
    client = TestClient(app)
    assert client.post("/v1/ask", json={"question": "q"}).status_code == 401
    headers = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/ask", json={"question": "q"}, headers=headers).status_code == 200
    assert client.post("/v1/ask", json={"question": "q"}, headers=headers).status_code == 200
    assert client.post("/v1/ask", json={"question": "q"}, headers=headers).status_code == 429


def test_deploy_artifacts(assistant: Assistant, tmp_path: Path) -> None:
    artifacts = assistant.deploy_artifacts(tmp_path / "deploy")
    names = {Path(f).name for f in artifacts.files}
    assert {"Dockerfile", "entrypoint.py", ".env.template", "requirements.lock"} <= names
    dockerfile = (tmp_path / "deploy" / "Dockerfile").read_text()
    assert "aire" in dockerfile


def test_project_builder_via_facade(tmp_path: Path) -> None:
    assistant = (
        AI.project("facade-demo")
        .documents(["OAuth2 bearer tokens are required for API access."])
        .model("mock:echo")
        .vector_store("local")
        .citations(True)
    )
    assistant.index()
    answer = assistant.ask("What tokens are required?")
    assert "OAuth2" in answer.text or "OAuth2" in answer.citations[0].excerpt
