import runpy

import pytest
from fastapi.testclient import TestClient

from aire import __version__
from aire.core.types import HealthStatus
from aire.deployment.artifacts import generate_artifacts
from aire.deployment.fastapi_app import create_app
from aire.models.builtin import EchoModel


def test_configured_backend_failure_does_not_become_echo(tmp_path, monkeypatch):
    generate_artifacts(tmp_path)
    monkeypatch.setenv("AIRE_DATABASE_URL", "postgresql://unavailable")
    import aire.rag.pgvector

    def unavailable(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(aire.rag.pgvector, "PgVectorStore", unavailable)
    app = runpy.run_path(str(tmp_path / "app.py"))
    with pytest.raises(RuntimeError, match="backend unavailable"):
        app["build_target"]()


def test_readiness_probes_backend_and_uses_current_version():
    class Unavailable(EchoModel):
        async def health(self):
            return HealthStatus.unhealthy("backend unavailable")

    with TestClient(create_app(Unavailable())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/openapi.json").json()["info"]["version"] == __version__


@pytest.mark.parametrize(
    "body", [{"question": "hello", "k": "bad"}, {"question": "hello", "k": -1}]
)
def test_invalid_retrieval_parameters_return_client_error(body):
    from aire.core.config import Settings
    from aire.core.runtime import Runtime
    from aire.models.base import run_sync
    from aire.rag.pipeline import Knowledge

    runtime = Runtime(Settings())
    with TestClient(create_app(Knowledge(runtime))) as client:
        response = client.post("/v1/ask", json=body)
        assert 400 <= response.status_code < 500
    run_sync(runtime.aclose())


def test_app_closes_only_explicitly_owned_runtime():
    from aire.core.config import Settings
    from aire.core.runtime import Runtime
    from aire.models.base import run_sync
    from aire.rag.pipeline import Knowledge

    runtime = Runtime(Settings())
    with TestClient(create_app(Knowledge(runtime))):
        pass
    assert not runtime._closed
    with TestClient(create_app(Knowledge(runtime), close_target=True)):
        pass
    assert runtime._closed
    run_sync(runtime.aclose())


def test_gateway_owns_runtime_only_when_requested():
    from aire.core.config import Settings
    from aire.core.runtime import Runtime
    from aire.deployment.gateway import create_gateway

    runtime = Runtime(Settings())
    with TestClient(create_gateway(runtime, models=["mock:echo"])) as client:
        assert client.get("/ready").status_code == 200
    assert not runtime._closed
    with TestClient(create_gateway(runtime, models=["mock:echo"], close_runtime=True)):
        pass
    assert runtime._closed


def test_service_agent_requests_do_not_share_conversation():
    from aire.agents.agent import Agent
    from aire.models.types import GenerationResult

    class Recorder(EchoModel):
        async def generate(self, request):
            users = [m.text_content for m in request.messages if m.role == "user"]
            return GenerationResult.text_result("|".join(users), model="test:recorder")

    agent = Agent(Recorder())
    with TestClient(create_app(agent)) as client:
        assert client.post("/v1/run", json={"input": "alice"}).json()["output"] == "alice"
        assert client.post("/v1/run", json={"input": "bob"}).json()["output"] == "bob"
    assert not agent.state.messages
