# MCP

aire implements a **subset** of the Model Context Protocol over **stdio** (newline-delimited JSON-RPC 2.0): tools, resources, and prompts. Not a full MCP host/feature set.

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

- Transport: stdio JSON-RPC subset only
- Surface focused on exposing aire `Tool`s (+ optional knowledge helpers)
- Do not assume sampling, roots, or every MCP capability from the full spec

See `AI.mcp.describe()` when present on your install for the live catalog.
