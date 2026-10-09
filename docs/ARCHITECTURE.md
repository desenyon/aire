# Architecture

aire is organized as a layered library under `src/aire/`. Heavy optional deps (torch, FastAPI, Neo4j, …) are imported only when a subsystem needs them.

## Package map

| Package | Role |
|---------|------|
| `core/` | Runtime, Settings, registries, plugins, errors, events |
| `models/` + `integrations/` | Model protocols + HTTP providers (`provider:name`) |
| `data/` | Loaders, Dataset, chunkers |
| `rag/` | Knowledge pipeline, stores, rewrite/compress/ACL/incremental |
| `graph/` | GraphRAG extract/store/communities |
| `agents/` | Deterministic agent runtime, builder, patterns, sessions, planning |
| `tools/` | `@tool`, builtins, OpenAPI import |
| `workflows/` | Graph engine + HITL |
| `evaluation/` | Metrics, runner, gates |
| `observability/` | Tracer, metrics, OTLP, analytics |
| `safety/` | Guardrails, redaction, PolicyEngine |
| `optimization/` | Cache, router, cost policy |
| `deployment/` | FastAPI app, OpenAI-compat gateway, artifacts, scale pack |
| `training/` | LoRA/quantize/distill/HPO + **toy** foundation stacks |
| `ml/` | Estimators / arch blocks (separate from agent story) |
| `mcp/` | Stdio MCP **subset** (tools/resources/prompts) |
| `cli/` | Typer entrypoint (`aire`) |
| `audio/`, `vision/`, `workers/`, … | Platform extras — several are stub-quality |

## Key decisions

### Offline-first

Builtins `mock:echo` and `local:hashing` implement the model/embedder contracts without network or downloads. CI and examples should prefer them.

### `provider:name` refs

Models, embedders, and stores resolve through `Ref.parse("provider:name")` and runtime registries. Examples: `mock:echo`, `openai:gpt-4o-mini`, `local:hashing`, `local:default` (vector store).

```python
model = AI.models.use_sync("mock:echo")
embedder = AI.models.embedder_sync("local:hashing")
store = AI.rag.vector_store("local:default")
```

### `.describe()`

Subsystems and many objects return JSON-serializable manifests via `.describe()`. Prefer this over reading source when exploring capabilities.

```python
AI.describe()
AI.agents.describe()
AI.models.describe()
```

### Deterministic agents

`Agent` runs a budgeted state machine (`AgentExecutor`): model call → optional tool calls → finish. Steps are recorded on `AgentState` / `AgentResult`. Planning (`AgentConfig.planning` / `PlanActVerify`) is an optional outer loop, not a free-form planner.

### Facade levels

1. **Declarative** — `AI.project(...)` / `AI.from_config("aire.yaml")`
2. **Composable** — `AI.models.use(...)`, `AI.rag.create(...)`, `AI.agents.create(...)`
3. **Low-level** — protocols and adapters directly

## Reliability boundaries

`Message.tool_calls` and `GenerationResult.tool_calls` share a provider-neutral
`ToolCall` type in `core/tool_calls.py` (re-exported from `models.types`). Provider
payload converters carry declarations across turns; streaming adapters buffer raw
argument fragments until a complete call can be validated.

`Knowledge._prepare_documents` performs loading/chunking/embedding preparation
before store publication. `VectorStore.replace_document` is a best-effort adapter
fallback; Local and SQLite override it. `replace_all` requires an atomic override.
SQLite commits database writes before changing its in-memory search cache.

Snapshot writers publish complete UTF-8 files with same-directory atomic replace.
This protects readers from partial JSON, not from concurrent writer conflicts.
Queue claims remain separate from results until acknowledgement. File workers use
local filesystem renames, Redis workers use reliable list moves and Lua; both need
explicit recovery after workers stop and idempotent side effects.

Workflow checkpoints persist consumed/fired edges and pending waves alongside
outputs and records. Skipped branches propagate to a fixed point. Resume retains
visit budgets and does not infer success from an exhausted failed node. See
[migration](migration.md) for old checkpoint and remote store limitations.
