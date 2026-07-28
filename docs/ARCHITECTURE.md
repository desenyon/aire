# aire — Architecture

## Layered design

```
┌─────────────────────────────────────────────────────────────────┐
│  Facade: aire.AI  (namespaces, AI.project builder, AI.from_config)│
├─────────────────────────────────────────────────────────────────┤
│  Capability modules                                             │
│  data · rag · tools · agents · workflows · evaluation · safety  │
│  optimization · multimodal · synthetic · training · deployment  │
├─────────────────────────────────────────────────────────────────┤
│  Universal interfaces                                           │
│  Model · EmbeddingModel · VectorStore · Chunker · Reranker ·    │
│  Tool · Memory · Guardrail · Trainer · Metric · Converter       │
├─────────────────────────────────────────────────────────────────┤
│  Core runtime                                                   │
│  Settings · Registry · PluginManager · EventBus · ResourceMgr · │
│  ExecutionContext · errors · content · serialization · logging  │
├─────────────────────────────────────────────────────────────────┤
│  Provider integrations (optional, lazily imported)              │
│  openai · anthropic · ollama · huggingface · qdrant · chroma    │
└─────────────────────────────────────────────────────────────────┘
```

Dependency rule: arrows point **downward only**. Core never imports capability
modules or providers; capability modules never import provider SDKs; providers
depend only on core + interfaces. The `aire.integrations.http.ProviderHttpClient`
is the single HTTP plumbing point all REST providers share.

## Key architectural decisions (ADRs, condensed)

### ADR-1: Content-addressed references (`provider:name`)

Every model, embedder, vector store, chunker and reranker is identified by a
`Ref` string such as `openai:gpt-4o-mini` or `local:default`. Resolution goes
through a `Registry[T]` populated by plugins. This makes provider switching a
configuration change, not a code change.

### ADR-2: Runtime as composition root

`aire.core.runtime.Runtime` wires settings, registries, the plugin manager,
event bus, resource manager and tracer. Everything that needs infrastructure
receives a `Runtime` (constructor injection). A lazily-created process-wide
default runtime backs the `AI` facade; tests and multi-tenant apps construct
explicit runtimes.

### ADR-3: Library-owned request/response types

Providers never leak vendor payloads. `GenerationRequest`, `GenerationResult`,
`EmbeddingRequest`, `Message` and the multimodal `*Content` blocks are owned by
`aire.models.types` / `aire.core.content`. Provider adapters translate at the
boundary. This is what makes `AI.models.use("openai:...")` →
`AI.models.use("ollama:...")` a one-line change.

### ADR-4: Deterministic agent state machine

Agents are explicit state machines (`AgentExecutor`), not recursive loops:

```
RUNNING → MODEL_CALL → PERMISSION_DENIED? → TOOL_CALL → OBSERVATION → FINISH
                      ↘ MAX_STEPS | BUDGET_EXCEEDED | FAILED
```

Every transition is recorded as an `AgentStep` with usage, so executions are
auditable, replayable (via `AgentState`) and budget-bounded
(`ExecutionContext.budget` enforced by `ctx.tick()`).

### ADR-5: Graph workflows with visit-bounded loops

The workflow engine (`aire.workflows.graph.Workflow`) schedules nodes by
edge-firing counts: a node runs when a new in-edge firing is available and all
predecessors are terminal. Conditional edges simply don't fire; unreachable
nodes are marked `SKIPPED`. Cycles are legal but bounded by `max_visits`,
giving loops without unbounded recursion. Checkpoints write the full
`WorkflowState` as JSON after every node transition.

### ADR-6: Structured errors as API

All failures are `AireError` subclasses carrying `code`, `message`, `context`,
`retryable` and a `cause` chain. Provider HTTP failures are mapped once in
`ProviderHttpClient`/`map_http_error` so callers always see
`RateLimitError`, `AuthenticationError`, `TimeoutError`, … regardless of vendor.
`wrap_errors` preserves unknown exceptions by wrapping them in
`InternalError` with the original attached.

### ADR-7: Plugins via entry points + programmatic registration

External packages register under the `aire.providers` entry-point group and
expose `register(runtime) -> PluginInfo`. Discovery is lazy: importing `aire`
never imports plugins; `PluginManager.discover()` runs only when a runtime is
built or `AI.plugins.discover()` is called. Built-in providers (`mock`,
`echo`, `local` stores) are registered directly by the runtime.

### ADR-8: Observability as a first-class seam

`Tracer`/`Span` mirror the OpenTelemetry shape (trace/span ids, attributes,
status) with `MemoryExporter` and `JsonlExporter` built in and sensitive
attributes masked. `Metrics` provides counters/gauges/latencies. The
`EventBus` broadcasts domain events (`agent.tool_call`, `model.generate`, …)
for audit and UI consumption. Tracing is wired through models, RAG, agents
and deployment endpoints — not bolted on per-call-site.

### ADR-9: Optional dependencies are hard-isolated

`fastapi`, `numpy`, `torch`, `datasets`, `pillow` are extras. Import sites are
either module-level `try/except ImportError` (deployment) or function-local
imports (training adapters), so `import aire` stays fast and dependency-light.
The performance test suite asserts import-time and dependency boundaries.

### ADR-10: Sync facade over async core

All network/inference code is async. For scripts and notebooks, thin `*_sync`
wrappers (`run_sync`) exist on the facade and builders; they refuse to run
inside an active event loop instead of deadlocking.

## Module map

```
src/aire/
  core/          runtime plumbing (no AI deps)
  models/        Model/EmbeddingModel ABCs, registry, builtins, retry
  integrations/  provider adapters (httpx-based) + plugin shims
  data/          Dataset, loaders, chunkers
  rag/           VectorStore, Retriever (hybrid + RRF), Reranker, Knowledge
  tools/         @tool decorator, ToolRegistry, builtin tools
  agents/        AgentExecutor state machine, Memory, Agent facade
  workflows/     graph engine, checkpoints, streaming events
  evaluation/    metrics registry, judges, Evaluator, reports
  observability/ tracing, metrics
  safety/        guardrails, redaction, approval policy, patterns
  optimization/  exact/semantic model cache, ModelRouter
  multimodal/    converter registry + model-backed conversions
  vision/ audio/ high-level pipelines over capable models
  synthetic/     model-driven dataset generation
  training/      framework-agnostic trainer loop
  deployment/    FastAPI factory, artifact generation
  cli/           typer CLI (aire ...)
  ai.py          AI facade namespaces
  knowledge_assistant.py  fluent AI.project builder (vertical slice)
```

## Testing architecture

- `tests/unit` — isolated behavior, no network.
- `tests/contract` — every Model/EmbeddingModel/VectorStore satisfies the same
  protocol-level contract.
- `tests/integration` — cross-module flows, including the offline vertical
  slice (ingest → ask → evaluate → trace → FastAPI deploy) and httpx-mocked
  provider payloads.
- `tests/security` — injection, path traversal, unsafe YAML, secret redaction,
  permission bypass.
- `tests/performance` — import time, embedding throughput, search latency,
  workflow overhead budgets.
