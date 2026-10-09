"""Exercise the actual HTTP adapter/agent boundary with a mock HTTP transport."""

import json

import httpx
import pytest

from aire.agents.agent import Agent
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.integrations.http import ProviderHttpClient
from aire.integrations.openai import OpenAIModel
from aire.models.base import run_sync
from aire.tools.tool import tool


@pytest.mark.integration
def test_provider_tool_loop_over_http_without_network(tmp_path):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "sum-1",
                                        "type": "function",
                                        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        history = body["messages"]
        assert history[-2]["tool_calls"][0]["id"] == "sum-1"
        assert history[-1]["tool_call_id"] == "sum-1"
        assert history[-1]["content"] == "5"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "five"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    @tool
    def add(a: int, b: int) -> int:
        """Add integers."""
        return a + b

    async def execute():
        async with Runtime(Settings()) as runtime:
            client = ProviderHttpClient(runtime, "test", base_url="https://provider.invalid")
            await client.raw.aclose()
            client._client = httpx.AsyncClient(
                base_url="https://provider.invalid", transport=httpx.MockTransport(handle)
            )
            try:
                agent = Agent(
                    OpenAIModel("test", client), tools=[add], session=tmp_path / "session.json"
                )
                result = await agent.run("Add two and three")
                assert result.ok and result.output == "five"
                assert result.usage.input_tokens == 10
                session = agent.session.load()
                assert session.messages[-1]["content"][0]["text"] == "five"
                assert session.messages[-3]["tool_calls"][0]["id"] == "sum-1"
            finally:
                await client.raw.aclose()

    run_sync(execute())
    assert len(requests) == 2
