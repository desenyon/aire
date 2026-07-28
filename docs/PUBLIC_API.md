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
| `AI.tools` | `tool(...)` decorator, `registry()`, `builtins()` |
| `AI.agents` | `create(model, tools=..., memory=..., config=...)`, `create_sync(...)` |
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

## RAG

```python
from aire.rag import (
    Knowledge, Document, Chunk, ScoredChunk, Citation, Answer, IndexReport,
    VectorStore, LocalVectorStore, Retriever, get_reranker,
    IdentityReranker, LexicalOverlapReranker,
)
```

## Tools & agents

```python
from aire.tools import tool, Tool, ToolRegistry, ToolSpec, ToolResult,
from aire.tools import SideEffect, RetryPolicy, builtin_tools
from aire.agents import Agent, AgentConfig, AgentStatus, AgentStep, AgentResult,
from aire.agents import Memory, BufferMemory, JsonlMemory
```

## Workflows

```python
from aire.workflows import Workflow, WorkflowState, WorkflowResult, WorkflowEvent, NodeSpec
```

`wf.add(name, fn, retries=..., timeout_seconds=..., requires_approval=...)`,
`wf.connect(a, b, when=...)`, `await wf.run(input)`, `wf.run_stream(input)`,
`wf.checkpoint(path)`, `wf.resume(state)`.

## Evaluation & observability

```python
from aire.evaluation import Evaluator, EvalCase, EvalReport, CaseResult, get_metric
from aire.observability import Tracer, Span, MemoryExporter, JsonlExporter, Metrics
```

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

- `create_gateway(runtime, models=..., aliases=..., embeddings=..., routing=..., objective=..., auth_token=..., rate_limit_per_minute=..., metrics=...)` —
  OpenAI-compatible gateway app (`/v1/chat/completions` with SSE streaming,
  `/v1/embeddings`, `/v1/models`, `/v1/gateway/manifest`).
- `Gateway(runtime, chat_routes=..., embedding_routes=..., routing=..., objective=...)` —
  transport-independent routing core; `.describe()` emits the gateway manifest.
- OpenAI-compatible provider aliases (registered lazily on first use):
  `lmstudio · llamacpp · llamafile · vllm · mlx · localai · tgi` (local),
  `groq · together · fireworks · deepseek · mistral · xai · openrouter · cerebras · perplexity` (hosted),
  and generic `openai_compatible:<model>` with `base_url=`.
- Config: `Settings.gateway` (`GatewayConfig`) — the `gateway:` section of `aire.yaml`.

CLI: `aire init · run · evaluate · serve · gateway · inspect · plugins · doctor · version`.

## Stability guarantees

- **Stable after 1.0**: everything above, plus the plugin contract in
  `docs/PLUGIN_SPEC.md` and error `code` values.
- **Provisional (may evolve pre-1.0 with changelog notes)**: training loop
  hooks, multimodal converter registry shape, deployment artifact templates.
- **Never public**: `aire.integrations.*` internals (construct through refs and
  the registry instead), any `_`-prefixed name, and test helpers.
