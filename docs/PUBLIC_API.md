# Public API

Symbols re-exported from `aire` (`src/aire/__init__.py`) and the `AI` facade (`src/aire/ai.py`).

## Top-level exports

| Symbol | Role |
|--------|------|
| `AI` | Unified facade |
| `Agent`, `AgentConfig`, `AgentResult`, `AgentStatus` | Agent types |
| `Team`, `TeamResult` | Multi-agent team |
| `tool`, `Tool`, `ToolSpec`, `ToolResult`, `SideEffect` | Tools |
| `Knowledge`, `Answer`, `Citation` | RAG |
| `Assistant` | Fluent project builder (`AI.project`) |
| `Model`, `EmbeddingModel`, `ModelInfo`, `GenerationRequest`, `GenerationResult` | Models |
| `Dataset`, `Record` | Data |
| `EvalCase`, `EvalReport`, `Evaluator` | Evaluation |
| `KnowledgeGraph` | GraphRAG |
| `LongTermMemory`, `MemoryEntry` | Memory |
| `Workflow`, `WorkflowResult` | Workflows |
| `Runtime`, `Settings` | Core runtime/config |
| `Message`, content types (`TextContent`, …) | Content |
| `Capability`, `HealthStatus`, `Manifest`, `Ref`, `Usage` | Shared types |
| `AireError`, `ConfigurationError`, `PermissionDeniedError`, `ProviderError` | Errors |
| `__version__` | Package version (`0.3.5`) |

Import what you need from `aire` first; deeper modules are stable for advanced use but not all are re-exported.

## `AI` facade namespaces

| Namespace | Highlights |
|-----------|------------|
| `AI.models` | `use` / `use_sync`, `embedder`, `router`, `cache`, `register_callable`, `describe` |
| `AI.data` | `load`, `chunker`, `describe` |
| `AI.rag` | `create` → `Knowledge`, `vector_store`, `incremental`, `describe` |
| `AI.graph` | `create`, `store`, `communities` |
| `AI.memory` | long-term memory helpers |
| `AI.mcp` | MCP server/client helpers |
| `AI.agents` | `create` / `create_sync`, `team`, `builder`, `pattern`, `toolkit`, `approver`, topologies |
| `AI.observe` | tracer, metrics, analytics, costs |
| `AI.deploy` | `api`, `artifacts`, `scale` |
| `AI.gateway` | `create`, `serve`, `endpoints` |
| `AI.workflows` | `create`, HITL helpers |
| `AI.training` | LoRA / quantize / distill / HPO / **foundation (toy)** |
| `AI.ml` | estimators / arch |
| `AI.safety` | `guardrails`, `redact`, `policy` |
| `AI.skills` | skill registry |
| `AI.schedule` | scheduler |
| `AI.workers` | in-process / file queue workers |
| `AI.recipes` | one-call scaffolds |
| `AI.locks` | project lock files |
| `AI.vision` / `AI.audio` / `AI.docs` | multimodal helpers (partially stubbed) |

## Class methods on `AI`

- `AI.runtime()` / `AI.configure(...)` — process-wide `Runtime`
- `AI.project(name)` / `AI.from_config(path)` → `Assistant`
- `AI.workflow(name)` / `AI.recipe(name)` / `AI.topologies()`
- `AI.tool(...)` — alias for registering tools
- `AI.evaluate(target, dataset, ...)` — evaluation runner
- `AI.describe()` — top-level manifest

## Typical usage

```python
from aire import AI, tool

model = AI.models.use_sync("mock:echo")

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = AI.agents.create_sync(model, tools=[add])
print(agent.run_sync("hello").output)
```
