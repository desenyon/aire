# aire for agents

aire is designed to be operated *by* agents, not just by humans. This document
is the entry point; the canonical, always-up-to-date guide ships **inside the
library** and is served over MCP.

## Read the guide through MCP

Every aire MCP server (`aire mcp-serve`) exposes knowledge resources and
prompt templates alongside tools:

| Method | Example | Returns |
| --- | --- | --- |
| `resources/list` | — | catalog of `aire://` documents |
| `resources/read` | `aire://guide` | the full usage guide (offline, packaged) |
| `resources/read` | `aire://manifest` | live `AI.describe()` (version, namespaces, registries) |
| `resources/read` | `aire://errors` | error taxonomy with retryability |
| `resources/read` | `aire://refs` | every `provider:name` scheme |
| `prompts/list` | — | task templates (`aire_rag`, `aire_agent`, `aire_gateway`, `aire_ml`, `aire_quickstart`) |
| `prompts/get` | `aire_rag` with `{"docs": "./manuals"}` | rendered instructions for the task |

From Python:

```python
client = await AI.mcp.connect(["aire", "mcp-serve"])
guide = await client.read_resource("aire://guide")
plan = await client.get_prompt("aire_agent", {"model": "openai:gpt-4o-mini"})
```

The same guide lives in the package at `aire/mcp/guide.md` — identical content
whether the library is installed from PyPI or a checkout.

## The operating contracts (summary)

1. **`provider:name` refs** address everything: models, embedders, vector
   stores, graph stores, estimators, memory.
2. **Discovery over assumption**: `.describe()` on any component,
   `AI.describe()` for the library, registry `.names()` for what's registered.
3. **Offline defaults for tests**: `mock:` models, `builtin:hash` embedders,
   `local:default` / `sqlite:memory` stores, `simple:*` estimators.
4. **Structured errors**: catch `AireError`, branch on `.code`, retry only
   when `.retryable`.
5. **Safety invariants**: never pickle; sync wrappers never inside a running
   event loop; budgets/permissions are runtime-enforced.

## Repo documentation map

- [PUBLIC_API.md](PUBLIC_API.md) — the supported public surface
- [ARCHITECTURE.md](ARCHITECTURE.md) — layers and invariants
- [PLUGIN_SPEC.md](PLUGIN_SPEC.md) — writing providers/plugins
- [SECURITY.md](SECURITY.md) — trust model and sandboxing
- [ROADMAP.md](ROADMAP.md) — where the library is going
