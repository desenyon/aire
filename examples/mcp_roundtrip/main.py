"""In-process MCP round-trip: tools/list + tools/call via MCPServer.handle."""

from __future__ import annotations

from aire import AI
from aire.models.base import run_sync
from aire.tools.builtins import builtin_tools


def main() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    server = AI.mcp.server([calc], knowledge=False)

    listed = run_sync(
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    )
    assert listed is not None
    tools = listed["result"]["tools"]
    print("tools:", [t["name"] for t in tools])

    called = run_sync(
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "calculator", "arguments": {"expression": "2+2"}},
            }
        )
    )
    assert called is not None
    print("call result:", called["result"])
    print("mcp:", AI.mcp.describe())


if __name__ == "__main__":
    main()
