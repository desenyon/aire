# MCP

aire implements a **subset** of the Model Context Protocol over **stdio** (newline-delimited JSON-RPC 2.0) and a Streamable HTTP client: tools, resources, and prompts. Not a full MCP host/feature set.

## Serve

```bash
aire mcp-serve
# equivalent
python -m aire.mcp
```

Embed:

```python
from aire.mcp import MCPServer
from aire.tools.builtins import builtin_tools

server = MCPServer(builtin_tools())
# await server.serve_stdio()
```

## Client

```python
from aire.mcp import MCPClient, MCPError
```

## Honesty

- Protocol version: `2025-06-18`. Server transport: stdio. Client transports: stdio and Streamable HTTP POST/SSE. No standalone GET event listener, resumable event store, elicitation, or full host implementation.
- Surface focused on exposing aire `Tool`s (+ optional knowledge helpers)
- Clients handle roots/list, configured sampling/createMessage callbacks, and progress notifications while awaiting responses. HTTP clients preserve session IDs and accept explicit authentication headers.
- Stdio requests have startup and per-request deadlines; close terminates the child and escalates to kill if needed. HTTP uses bounded network timeouts.
- Imported MCP tools default to external side effects and remain subject to approval policies.

See `AI.mcp.describe()` when present on your install for the live catalog.

## HTTP client

```python
from aire.mcp.http_client import MCPHttpClient

async def connect(url, token):
    async with MCPHttpClient(url, headers={"Authorization": f"Bearer {token}"}) as client:
        return await client.list_tools()
```

Generated MCP/OpenAPI argument validators cover objects, arrays, primitive types,
local references, enums, `anyOf`/type unions and scalar bounds. Recursive schemas, `oneOf`, typed additional properties and
conditional/composition keywords outside that subset fail explicitly. OpenAPI
imports use HTTP(S) bases and relative operation paths; nested JSON bodies,
path, query and header inputs are validated before calls. Inline skill JSON tools
return static reference text and identify that in their description; executable
skill tools belong in `tools.py`. Import failures are reported, never omitted.
