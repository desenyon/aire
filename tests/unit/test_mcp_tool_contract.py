import pytest

from aire.mcp.client import MCPClient
from aire.mcp.http_client import MCPHttpClient
from aire.models.base import run_sync


@pytest.mark.parametrize("client", [MCPClient(["unused"]), MCPHttpClient("http://unused")])
def test_remote_tool_schema_and_effects_are_enforced(client):
    tool = client._adapt(
        {
            "name": "write",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        }
    )
    assert str(tool.spec.side_effect) == "external_side_effect"
    result = run_sync(tool.execute({"value": "not an integer"}))
    assert not result.ok
    assert result.error_code == "tool.arguments_invalid"
