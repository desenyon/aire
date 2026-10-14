from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from aire.core.config import Settings
from aire.core.errors import ProviderError
from aire.core.runtime import Runtime
from aire.deployment.gateway import create_gateway
from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.models.types import GenerationChunk, GenerationRequest
from aire.safety.guardrails import GuardrailVerdict


class OutputRail:
    name = "output-test"

    def check(self, text):
        return GuardrailVerdict(passed="BLOCKME" not in text, guardrail=self.name)


class SplitModel(EchoModel):
    fail = False
    closed = False

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        try:
            yield GenerationChunk(text="BLOCK")
            if self.fail:
                raise ProviderError("test", "stream interrupted")
            yield GenerationChunk(text="ME", finish_reason="stop")
        finally:
            self.closed = True


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("fail", [False, True])
def test_stream_releases_no_unchecked_output(path, fail):
    runtime = Runtime(Settings())
    model = SplitModel()
    model.fail = fail
    runtime.model_providers.register("test", lambda name, **kw: model)
    app = create_gateway(runtime, models=["test:split"], guardrails=[OutputRail()])
    with TestClient(app) as client:
        response = client.post(
            path,
            json={
                "model": "test:split",
                "stream": True,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert "BLOCK" not in response.text
        assert "error" in response.text
    assert model.closed
    run_sync(runtime.aclose())


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_stream_redacts_email_spanning_chunks(path):
    from aire.safety.guardrails import PIIGuardrail

    class EmailModel(EchoModel):
        async def stream(self, request):
            yield GenerationChunk(text="person@")
            yield GenerationChunk(text="example.com", finish_reason="stop")

    runtime = Runtime(Settings())
    runtime.model_providers.register("test", lambda name, **kw: EmailModel())
    with TestClient(
        create_gateway(runtime, models=["test:email"], guardrails=[PIIGuardrail(action="redact")])
    ) as client:
        response = client.post(
            path,
            json={
                "model": "test:email",
                "stream": True,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert "person@" not in response.text
        assert "[PII]" in response.text
    run_sync(runtime.aclose())
