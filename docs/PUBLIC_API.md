# aire — Public API

This document defines the supported surface. Anything importable from a
package `__init__` (or documented here) is public; everything else —
especially modules and names prefixed with `_` — is internal and may change
without notice. From 1.0, breaking changes to the public API require a
deprecation cycle and a migration note.

## Top level

```python
from aire import AI, __version__
from aire.core.errors import AireError
```

## The AI facade

`AI` exposes lazily-bound namespaces over the default runtime:

| Namespace | Highlights |
|---|---|
| `AI.models` | `use(spec)`, `use_sync(spec)`, `embedder(spec)`, `register_callable(name, fn)`, `router(candidates, objective=...)`, `cache(model)` |
| `AI.data` | `load(source, ...)`, `from_texts(texts)`, `chunker(name, **opts)` |
| `AI.rag` | `knowledge(...)`, `vector_store(spec)` |
| `AI.graph` | `create(...)` (KnowledgeGraph), `store(spec)` (graph stores) |
| `AI.memory` | `create(path=..., embedder=...)` (LongTermMemory) |
| `AI.mcp` | `server(tools)`, `connect(command)`, `connect_sync(command)` |
| `AI.tools` | `tool(...)` decorator, `registry()`, `builtins()` |
| `AI.agents` | `create(model, tools=..., memory=..., config=...)`, `create_sync(...)`, `team(members, supervisor=...)` |
| `AI.workflows` | `create(name, ...)` |
| `AI.evaluate` | `run(target, dataset, metrics=...)` |
| `AI.observe` | `tracer()`, `metrics()` |
| `AI.safety` | `guardrails(*names)`, `redact(text)` |
| `AI.deploy` | `api(target, ...)`, `artifacts(target, dir)` |
| `AI.gateway` | `create(...)`, `serve(host, port, ...)`, `endpoints()` |
| `AI.project` | fluent builder (`aire.knowledge_assistant.Assistant`) |
| `AI.configure` | `configure(settings_or_path)`, `runtime()` |

## Core contracts

```python
from aire.core.content import (
    Message, TextContent, ImageContent, AudioContent, VideoContent,
    DocumentContent, StructuredContent,
)
from aire.core.config import Settings, load_settings
from aire.core.context import Budget, ExecutionContext
from aire.core.errors import (
    AireError, ConfigurationError, DataError, NotFoundError, ProviderError,
    AuthenticationError, RateLimitError, TimeoutError, BudgetExceededError,
    PermissionDeniedError, SafetyError, OutputValidationError, WorkflowError,
    RetrievalError, PluginError, InternalError, ensure_aire_error,
)
from aire.core.runtime import Runtime
from aire.core.types import Capability, HealthStatus, Manifest, Ref, Usage
```

## Models

```python
from aire.models import (
    Model, EmbeddingModel, ModelRegistry, register_callable,
    EchoModel, CallableModel, HashingEmbedder, with_retry,
    GenerationRequest, GenerationResult, GenerationChunk, EmbeddingRequest,
    EmbeddingResult, ModelInfo, ToolCall, ToolDefinition, StructuredOutputSpec,
)
```

- `Model.generate(request) -> GenerationResult`
- `Model.stream(request) -> AsyncIterator[GenerationChunk]`
- `Model.ask(prompt) -> str`, `Model.ask_structured(prompt, SchemaModel)`
- `EmbeddingModel.embed(request) -> EmbeddingResult`, `.embed_one(text)`
- `ModelRegistry(runtime).use("provider:name")`, `.embedder(spec)`

## Data

```python
from aire.data import (
    Dataset, Record, DatasetSplit, QualityReport, load,
    Chunker, FixedChunker, SentenceChunker, RecursiveChunker, TextChunk, get_chunker,
)
```

Chainable: `dataset.validate().deduplicate().split(train=0.8, validation=0.1, test=0.1)`.

`load(source)` accepts: `.jsonl`/`.json`/`.csv` files, `.html`/`.htm` (clean
text extraction, also for HTML URLs and directory members), `.parquet`/`.xlsx`/
`.xls` (lazy pandas, `aire[ml]`), text files, directories, `http(s)://` URLs,
and in-memory lists. `aire.data.loaders.html_to_text` is exported for direct use.

## RAG

```python
from aire.rag import (
    Knowledge, Document, Chunk, ScoredChunk, Citation, Answer, IndexReport,
    VectorStore, LocalVectorStore, Retriever, get_reranker,
    IdentityReranker, LexicalOverlapReranker,
)
```

Vector store refs: `local:*`, `sqlite:<path>` (embedded, transactional),
`qdrant:*`, `chroma:*`, `pinecone:*`, `weaviate:*` (native BM25), `milvus:*`.

## Knowledge graphs (GraphRAG)

```python
from aire.graph import (
    KnowledgeGraph, GraphStore, SQLiteGraphStore,
    GraphExtractor, LexicalGraphExtractor, ModelGraphExtractor,
    Entity, Relation, Extraction, Subgraph, GraphIndexReport,
)
```

- `await graph.ingest(source)` — chunk, extract triples, index chunks.
- `await graph.subgraph(question, depth=1)` — entity linking + BFS neighborhood.
- `await graph.query(question, k=5)` — graph + vector fused, cited `Answer`.
- Graph store refs via `AI.graph.store("sqlite:<path>")` (`sqlite:memory` default).

## Long-term memory

```python
from aire.memory import LongTermMemory, MemoryEntry, MemoryKind
```

Implements the agent `Memory` interface (drop-in `Agent(memory=...)`) plus
`remember()`, `recall_semantic(query, k=...)`, `consolidate(model)`, and
optional JSONL persistence (`path=`).

## MCP (Model Context Protocol)

```python
from aire.mcp import MCPServer, MCPClient, MCPError
```

- `MCPServer(tools, knowledge=True)` — `handle(message)`, `serve_stdio()`;
  CLI: `aire mcp-serve`. Exposes knowledge resources (`aire://guide`,
  `aire://manifest`, `aire://errors`, `aire://refs`) and task prompts
  (`aire_quickstart`, `aire_rag`, `aire_agent`, `aire_gateway`, `aire_ml`).
- `MCPClient(command)` — async context manager; `list_tools()`,
  `call_tool(name, args)`, `tools()` (adapts remote tools into aire `Tool`s),
  `list_resources()`, `read_resource(uri)`, `list_prompts()`,
  `get_prompt(name, args)`.
- Transport: newline-delimited JSON-RPC 2.0 over stdio (protocol `2025-06-18`).

## Model creation (ML)

```python
from aire.ml import (
    Estimator, FitReport, Prediction, TaskType,
    MajorityClassifier, CentroidClassifier, KNNClassifier, LinearRegressor,
    SklearnEstimator, TorchEstimator,
    frame_to_dataset, dataset_to_frame, predictions_to_frame, available_backends,
)
```

- `AI.ml.create(spec, **options)` / `AI.ml.fit(spec, dataset, target=)` /
  `AI.ml.backends()` / `AI.ml.catalog()` / `AI.ml.to_frame(ds)` /
  `AI.ml.from_frame(df, target=)`.
- `AI.ml.cross_validate(spec, dataset, k=)` / `AI.ml.grid_search(spec, dataset,
  param_grid)` / `est.feature_importance(dataset)`.
- Estimator refs: `simple:majority · centroid · knn · linear_regression`
  (native, offline), `sklearn:<name|dotted.path>` (aire[ml]),
  `torch:mlp` with `hidden=`, `module_factory=` (aire[torch]).
- Contract: `await est.fit(dataset, target=)` → `FitReport`;
  `await est.predict(records)` → `Prediction(value, probabilities)`;
  `await est.evaluate(dataset)` → classification report or MAE/RMSE/R²;
  `est.save(path)` / `est.load(path)`; `est.describe()`. Feature convention:
  `metadata["features"]` → numeric metadata → text-derived. aire never
  pickles: sklearn persists via `skops`/`joblib` on `est.model`; torch loads
  use `weights_only=True`.

## Tools & agents

```python
from aire.tools import tool, Tool, ToolRegistry, ToolSpec, ToolResult,
from aire.tools import SideEffect, RetryPolicy, builtin_tools
from aire.agents import Agent, AgentConfig, AgentStatus, AgentStep, AgentResult,
from aire.agents import Memory, BufferMemory, JsonlMemory
from aire.agents import Team, TeamResult, Delegation, DelegationRecord
```

- `agent.as_tool(name=..., description=...)` — agent-as-tool composition.
- `Team(members, supervisor, max_rounds=6)` — supervisor-routed delegation with
  structured decisions and auditable `DelegationRecord`s; `AI.agents.team(...)`.
- Approval policies (`aire.agents.approvals`, or `AI.agents.approver(kind)`):
  `RuleApprover(auto_approve_below=..., allow=..., deny=...)` with audit trail,
  `InteractiveApprover()` human-in-the-loop prompts with session memory.

## Workflows

```python
from aire.workflows import Workflow, WorkflowState, WorkflowResult, WorkflowEvent, NodeSpec
```

`wf.add(name, fn, retries=..., timeout_seconds=..., requires_approval=...)`,
`wf.connect(a, b, when=...)`, `await wf.run(input, state=...)`,
`wf.run_stream(input)`, `Workflow(checkpoint_path=...)` (state persisted after
every node), `Workflow.load_checkpoint(path)`, `await wf.resume(path?)`
(continues from a checkpoint: completed nodes are skipped, failed nodes retry).

## Evaluation & observability

```python
from aire.evaluation import Evaluator, EvalCase, EvalReport, CaseResult, get_metric
from aire.observability import Tracer, Span, MemoryExporter, JsonlExporter, Metrics
from aire.observability import OTLPExporter  # batched OTLP/HTTP+JSON to any collector
```

- `FunctionTrainer(step, config)` — `await trainer.fit(dataset, resume_from=...)`
  (continues from a checkpoint), `FunctionTrainer.load_checkpoint(path)`.

## Safety & optimization

```python
from aire.safety import (
    Guardrail, GuardrailChain, PIIGuardrail, InjectionGuardrail, SecretGuardrail,
    ApprovalPolicy, redact, redact_pii, redact_secrets,
)
from aire.optimization import CachedModel, SemanticCachedModel, ModelRouter, RouteDecision
```

## Deployment & CLI

```python
from aire.deployment import Gateway, create_app, create_gateway, generate_artifacts
```

- `create_gateway(runtime, models=..., aliases=..., embeddings=..., routing=..., objective=..., budgets=..., circuit_breaker=..., failure_threshold=..., cooldown_seconds=..., request_log=..., auth_token=..., rate_limit_per_minute=..., metrics=...)` —
  OpenAI-compatible gateway app (`/v1/chat/completions` with SSE streaming,
  Anthropic-compatible `/v1/messages`, `/v1/embeddings`, `/v1/models`,
  `/v1/gateway/manifest`). Circuit breakers skip failing candidates; daily cost
  budgets cap spend per alias/ref; `request_log` writes a JSONL audit trail.
- `Gateway(runtime, chat_routes=..., embedding_routes=..., routing=..., objective=...)` —
  transport-independent routing core; `.describe()` emits the gateway manifest
  (routes, circuit states, budgets, today's spend).
- OpenAI-compatible provider aliases (registered lazily on first use):
  `lmstudio · llamacpp · llamafile · vllm · mlx · localai · tgi` (local),
  `groq · together · fireworks · deepseek · mistral · xai · openrouter · cerebras · perplexity` (hosted),
  and generic `openai_compatible:<model>` with `base_url=`.
- Config: `Settings.gateway` (`GatewayConfig`) — the `gateway:` section of `aire.yaml`.

CLI: `aire init · run · evaluate · serve · gateway · mcp-serve · inspect · plugins · doctor · version`.

## Stability guarantees

- **Stable after 1.0**: everything above, plus the plugin contract in
  `docs/PLUGIN_SPEC.md` and error `code` values.
- **Provisional (may evolve pre-1.0 with changelog notes)**: training loop
  hooks, multimodal converter registry shape, deployment artifact templates.
- **Never public**: `aire.integrations.*` internals (construct through refs and
  the registry instead), any `_`-prefixed name, and test helpers.
