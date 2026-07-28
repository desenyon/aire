# aire

**The agent-first AI creation library.** One consistent interface from idea to deployed AI system — models, data, RAG, agents, tools, workflows, evaluation, safety, observability, and deployment, all composable, all inspectable, all provider-independent.

```python
from aire import AI

assistant = (
    AI.project("knowledge_assistant")
    .documents("./docs")
    .model("openai:gpt-4o-mini")  # or mock:echo, ollama:llama3.2, anthropic:claude-sonnet-4-5
    .vector_store("local:default")
    .citations(True)
)

assistant.index()

answer = assistant.ask("What does the documentation say about authentication?")
print(answer.text)
print(answer.citations)

assistant.evaluate("./evals.jsonl", metrics=["accuracy", "groundedness"])
app = assistant.deploy()  # production FastAPI app with /health, /ready, /manifest, /metrics
```

**Works fully offline out of the box** (`mock:echo` model + `local:hashing` embedder) — every subsystem, example, and test runs with zero credentials and zero network. Swapping providers is a one-string change.

---

## Table of contents

- [Why agent-first](#why-agent-first)
- [The problems aire fixes](#the-problems-aire-fixes)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [The AI facade](#the-ai-facade)
- [Models](#models)
- [Data](#data)
- [Retrieval augmented generation](#retrieval-augmented-generation)
- [Knowledge graphs and GraphRAG](#knowledge-graphs-and-graphrag)
- [Model creation (ML)](#model-creation-ml)
- [Tools](#tools)
- [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [Agents](#agents)
- [Long-term memory](#long-term-memory)
- [Multi-agent teams](#multi-agent-teams)
- [Workflows](#workflows)
- [Evaluation](#evaluation)
- [Observability](#observability)
- [Safety](#safety)
- [Optimization](#optimization)
- [Multimodal, vision and audio](#multimodal-vision-and-audio)
- [Synthetic data](#synthetic-data)
- [Training](#training)
- [Deployment](#deployment)
- [Model gateway](#model-gateway)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Plugins and providers](#plugins-and-providers)
- [Examples](#examples)
- [Documentation map](#documentation-map)
- [Development](#development)
- [License](#license)

---

## Why agent-first

aire is designed so that **both humans and coding agents** can build production AI systems on it without reading its source code:

- **Everything is discoverable.** Every component — model, tool, workflow, vector store — emits a machine-readable `.describe()` manifest with schemas, capabilities and permissions. An agent can enumerate what exists and what it accepts at runtime.
- **Everything is a tool.** `@AI.tool` turns any Python function into a self-describing, permissioned, rate-limited, auditable tool with JSON-schema contracts derived from its signature.
- **Deterministic agent runtime.** Agents execute as explicit state machines (model call → permission check → tool call → observation → finish) with token/cost/step budgets — never unbounded recursion. Every transition is a recorded `AgentStep`.
- **Structured everything.** All requests, responses and configuration are Pydantic models. All failures are `AireError` subclasses with stable machine-readable `code`, `context`, and `retryable` flags.
- **Provider independence.** `provider:name` references (`openai:gpt-4o-mini`, `anthropic:claude-sonnet-4-5`, `ollama:llama3.2`) resolve through a plugin registry. The core never imports a vendor SDK — providers are plain HTTP adapters.
- **Inspectable by default.** OpenTelemetry-shaped tracing, metrics, event history, evaluation reports, and workflow checkpoints are built in, not bolted on.

See [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) for the full product principles and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the ten architectural decisions behind this design.

## The problems aire fixes

Building AI systems today means fighting the same structural problems over and over. aire exists to fix them at the library level:

| Problem in AI creation today | How aire fixes it |
|---|---|
| **Provider lock-in.** Every vendor SDK has its own payload shapes, error formats and streaming quirks; swapping providers means rewriting application code. | One normalized `Model` / `EmbeddingModel` interface. Providers are thin HTTP adapters behind `provider:name` refs — swapping OpenAI for a local GGUF model is a one-string change. |
| **Local models are second-class.** GGUF, MLX and self-hosted servers all speak slightly different dialects and are painful to wire into agent stacks. | First-class refs for every local server — `llamacpp:`, `lmstudio:`, `vllm:`, `mlx:`, `localai:`, `llamafile:`, `tgi:`, `ollama:` — plus `openai_compatible:` with a custom `base_url` for anything else. |
| **No built-in routing or gateway.** Teams hand-roll proxies for fallbacks, round-robin and cost-based routing, then again for auth, rate limits and observability. | `AI.gateway.create()` / `aire gateway` serves an OpenAI-compatible endpoint in front of any refs, with fallback chains, round-robin or objective routing, bearer auth, rate limiting, streaming, tracing and metrics built in. |
| **Frameworks are hostile to coding agents.** Magic globals, stringly-typed configs and hidden state make agent-driven development unreliable. | Agent-first contracts: `.describe()` manifests, Pydantic-everything, structured `AireError` codes, deterministic agent state machines with explicit budgets and permissions. |
| **Production concerns arrive too late.** Tracing, cost accounting, evaluation, guardrails and deployment get bolted on after the demo works — if ever. | They are core subsystems: observability, evaluation, safety and FastAPI deployment ship in the box and work offline from the first line of code. |
| **Nothing runs without credentials.** Most libraries can't even be tested without an API key, so CI and local development depend on live vendors. | `mock:echo` + `local:hashing` run the entire stack — agents, RAG, workflows, evals, gateway — fully offline. |

## Installation

```bash
pip install aire                 # core — zero heavy dependencies
pip install "aire[serve]"        # + FastAPI/uvicorn for deployment
pip install "aire[numpy]"        # + numpy acceleration
pip install "aire[datasets]"     # + Hugging Face datasets loader
pip install "aire[training]"     # + PyTorch training adapters
pip install "aire[vision]"       # + Pillow for image pipelines
pip install "aire[all]"          # everything
pip install "aire[dev]"          # development tooling (ruff, mypy, pytest)
```

Requires **Python 3.11+**. Core dependencies are only `pydantic`, `httpx`, `pyyaml`, and `typer`; provider integrations share one `httpx` plumbing layer, so optional extras stay truly optional.

## Quick start

```bash
pip install aire
aire init my-project && cd my-project   # scaffold a project (aire.yaml + app.py)
aire doctor                             # check environment, config, provider access
aire run "hello, aire"                  # one-shot generation (offline by default)
python app.py                           # a complete offline RAG pipeline
```

From Python, three levels of abstraction are always available:

```python
from aire import AI

# 1. Declarative — everything from config
app = AI.from_config("aire.yaml")

# 2. Composable — wire subsystems yourself
model = AI.models.use_sync("openai:gpt-4o-mini")
knowledge = AI.rag.create()
agent = AI.agents.create_sync(model, tools=["calculator", "read_file"])

# 3. Low level — direct protocol access
from aire.models.types import GenerationRequest

result = await model.generate(GenerationRequest.of("hello"))
```

## Core concepts

### References: `provider:name`

Every resolvable component is identified by a `Ref` string:

```
openai:gpt-4o-mini        anthropic:claude-sonnet-4-5     ollama:llama3.2
huggingface:meta-llama    mock:echo                       callable:my_function
local:default             qdrant:my-collection            chroma:docs
```

The left side selects the provider factory from a registry; the right side is passed to it. Changing providers never changes application code.

### The Runtime

`aire.core.runtime.Runtime` is the composition root: it holds settings, component registries, the plugin manager, the event bus, the resource manager, and a tracer. The `AI` facade operates on a lazily-created process-wide default runtime; tests and multi-tenant apps construct their own:

```python
from aire.core.runtime import Runtime

runtime = Runtime.from_config("aire.yaml")
await runtime.aclose()  # closes every tracked resource, LIFO
```

### Manifests

Every public component implements `.describe()`, returning a `Manifest` (kind, name, provider, capabilities, input/output/config schemas). This is the discovery contract agents use to explore the library.

### Structured errors

```python
from aire.core.errors import AireError

try:
    answer = await knowledge.ask("...")
except AireError as exc:
    exc.code  # e.g. "provider.rate_limited" — stable, machine-readable
    exc.retryable  # can this be retried?
    exc.context  # structured details (provider, path, missing permissions, ...)
    exc.to_dict()  # JSON-safe payload for APIs and logs
```

The hierarchy includes `ProviderError`, `RateLimitError`, `AuthenticationError`, `TimeoutError`, `NotFoundError`, `BudgetExceededError`, `PermissionDeniedError`, `SafetyError`, `OutputValidationError`, `WorkflowError`, `DataError`, `ConfigurationError`, and `PluginError`. Provider HTTP failures are mapped once in the shared HTTP client, so every vendor looks identical to callers.

### Multimodal content

Normalized content primitives flow through the whole library:

```python
from aire.core.content import TextContent, ImageContent, AudioContent, StructuredContent

result = await model.generate(
    GenerationRequest(
        messages=[
            Message(
                role="user",
                content=[
                    TextContent(text="Analyze this diagram"),
                    ImageContent.from_file("diagram.png"),
                ],
            )
        ]
    )
)
```

## The AI facade

`AI` is the single entry point. Each namespace is lazily bound to the default runtime:

| Namespace | Purpose |
|---|---|
| `AI.models` | Resolve/route/cache models and embedders |
| `AI.data` | Load and transform datasets; chunkers |
| `AI.rag` | Knowledge pipelines and vector stores |
| `AI.graph` | Knowledge graphs and GraphRAG pipelines |
| `AI.memory` | Long-term agent memory (episodic + semantic) |
| `AI.ml` | Estimators + Pipeline/Transform + arch/optim/loss |
| `AI.mcp` | MCP servers and clients (Model Context Protocol) |
| `AI.tool` / `AI.tools()` | Tool decorator / runtime tool registry |
| `AI.agents` | Create agents and supervisor-routed teams |
| `AI.workflows` | Graph workflow engine |
| `AI.evaluate(...)` | Evaluate any target, synchronously |
| `AI.observe` | Tracer, metrics, trace/event history |
| `AI.safety` | Guardrail chains, redaction |
| `AI.training` | Trainer factory |
| `AI.synthetic(...)` | Synthetic data generator factory |
| `AI.deploy` | FastAPI apps and deployment artifacts |
| `AI.gateway` | OpenAI-compatible model gateway (routing, fallbacks, auth) |
| `AI.project` | Fluent builder (the flagship experience) |
| `AI.configure` | Replace the default runtime/settings |

The full, stability-guaranteed surface is enumerated in [docs/PUBLIC_API.md](docs/PUBLIC_API.md).

## Models

```python
from aire import AI

model = await AI.models.use("openai:gpt-4o-mini")
text = await model.ask("summarize aire in one sentence")

# Streaming
async for chunk in model.stream(GenerationRequest.of("tell me a story")):
    print(chunk.text, end="")

# Structured output — validated, retried
from pydantic import BaseModel


class Summary(BaseModel):
    title: str
    points: list[str]


summary = await model.ask_structured("summarize the refund policy", Summary)

# Tool calling: pass tool definitions, get ToolCall objects back
result = await model.generate(GenerationRequest.of("add 2 and 3", tools=registry.definitions()))

# Embeddings
embedder = await AI.models.embedder()  # default: local:hashing (offline)
vectors = await embedder.embed(EmbeddingRequest(texts=["hello", "world"]))
```

Every model exposes normalized `ModelInfo` metadata (context window, capabilities, streaming/tool/structured-output support, cost rates) via `.info`, plus `.health()` and `.describe()`.

**Any Python function can be a model:**

```python
AI.models.register_callable("uppercase", lambda prompt: prompt.upper())
model = await AI.models.use("callable:uppercase")
```

Built-in offline implementations (`mock:echo`, `local:hashing`) make every test and example runnable with no credentials.

### Local models and OpenAI-compatible endpoints

Any server that speaks the OpenAI chat-completions protocol works out of the box — no per-vendor adapter, no vendor SDK. Named aliases carry sensible default base URLs and env vars:

```python
# Local inference servers — no API key needed
model = AI.models.use_sync("lmstudio:qwen2.5-7b-instruct")  # LM Studio (GGUF/MLX)
model = AI.models.use_sync("llamacpp:llama-3.2")  # llama.cpp server (GGUF)
model = AI.models.use_sync("vllm:meta-llama/Llama-3.1-8B")  # vLLM
model = AI.models.use_sync("mlx:mlx-community/Llama-3.2-3B")  # mlx-lm (Apple Silicon)
model = AI.models.use_sync("ollama:llama3.2")  # Ollama

# Hosted OpenAI-compatible APIs — key from env (GROQ_API_KEY, MISTRAL_API_KEY, ...)
model = AI.models.use_sync("groq:llama-3.3-70b-versatile")
model = AI.models.use_sync("deepseek:deepseek-chat")
model = AI.models.use_sync("mistral:mistral-large-latest")
model = AI.models.use_sync("openrouter:anthropic/claude-sonnet-4-5")

# Anything else with an OpenAI-compatible endpoint
model = AI.models.use_sync(
    "openai_compatible:my-finetune",
    base_url="http://gpu-box:8000/v1",
    api_key="...",
)
```

Full catalog (including `localai`, `llamafile`, `tgi`, `together`, `fireworks`, `xai`, `cerebras`, `perplexity`): `AI.gateway.endpoints()` — machine-readable for agents.

## Data

```python
from aire import AI

dataset = AI.data.load("./support_data")  # file, directory, URL, JSONL/CSV/JSON/text
dataset = (
    dataset.validate(min_length=10)  # quality gates with a QualityReport
    .deduplicate()  # content-hash based
    .filter(lambda r: "refund" in r.text)
    .map(lambda r: r.model_copy(update={"text": r.text.strip()}))
)
split = dataset.split(train=0.8, validation=0.1, test=0.1, seed=42)
dataset.to_jsonl("clean.jsonl")  # round-trips losslessly

report = dataset.quality_report()  # counts, lengths, PII hints, duplicates
```

Chunkers prepare text for embedding:

```python
chunker = AI.data.chunker("recursive", chunk_size=512, overlap=64)  # fixed | sentence | recursive
chunks = chunker.chunk("long document text ...")
```

## Retrieval augmented generation

```python
from aire import AI

knowledge = AI.rag.create()  # or via AI.project(...) fluent builder
report = await knowledge.ingest("./documents")  # load → chunk → embed → store
answer = await knowledge.ask(
    "What are the refund rules?",
    model="openai:gpt-4o-mini",
    k=5,
    citations=True,
)
print(answer.text)
for c in answer.citations:
    print(c.source, c.score)
```

Under the hood: pluggable **chunker** → **embedder** → **vector store** (`local:default`, embedded `sqlite:<path>`, `qdrant:*`, `chroma:*`, `pinecone:*`, `weaviate:*`, `milvus:*`) → **hybrid retriever** (vector similarity + keyword, fused with reciprocal rank) → **reranker** (`lexical` by default) → grounded prompt with numbered citations.

## Knowledge graphs and GraphRAG

Vectors find *similar* text; graphs find *connected* facts. `AI.graph` builds a knowledge graph from your documents and answers from it — with the same cited `Answer` contract as classic RAG:

```python
from aire import AI

graph = AI.graph.create()  # embedded sqlite triple store + lexical extractor
report = await graph.ingest("./documents")  # chunk → extract triples → store

facts = await graph.subgraph("How do refunds relate to chargebacks?")
print(facts.as_context())  # Refunds —governed_by→ Policy, ...

answer = await graph.query("How do refunds relate to chargebacks?")
print(answer.text, answer.citations)
```

Two extractor strategies: the default `LexicalGraphExtractor` is deterministic and **works fully offline** (capitalized phrases as entities, sentence co-occurrence as relations); pass any model ref for typed, semantic triples — `AI.graph.create(model="ollama:llama3.2")`. Querying links question terms to entities, expands their BFS neighborhood, fuses graph facts with vector retrieval, and grounds the answer in both. The graph store is `sqlite:<path>` — single-file, transactional, stdlib-only; the `GraphStore` interface is pluggable for graph databases.

## Model creation (ML)

aire orchestrates the ML ecosystem instead of reimplementing it. One
`Estimator` / `Transform` / `Pipeline` contract — `fit(dataset, target=)` →
`predict` → `evaluate` → `save`/`load` — spans native zero-dependency learners,
scikit-learn, PyTorch, Keras, XGBoost, and LightGBM, all addressed by
`backend:name` refs:

```python
from aire import AI

AI.ml.backends()  # native/sklearn/torch/keras/xgboost/lightgbm/catboost/pandas/polars

est = await AI.ml.fit("simple:centroid", dataset)  # offline, zero deps
est = await AI.ml.fit("sklearn:random_forest", dataset, n_estimators=100)  # aire[ml]
est = await AI.ml.fit("torch:mlp", dataset, hidden=(64, 32), epochs=300,
                      optimizer="adamw", validation_split=0.2, amp=False)  # aire[torch]
est = await AI.ml.fit("keras:mlp", dataset, metrics=["accuracy"],
                      callbacks=["early_stopping"])  # aire[keras]
est = await AI.ml.fit("catboost:classifier", dataset)  # aire[catboost]

ct = AI.ml.column_transformer([("num", "native:standard_scaler", ["x", "y"])])
pipe = AI.ml.pipeline([
    ("scale", "native:standard_scaler"),
    ("clf", "simple:centroid"),
])
await pipe.fit(dataset)

AI.ml.scorers()  # accuracy, roc_auc, log_loss, …
cv = await AI.ml.cross_validate("simple:centroid", dataset, k=5, stratified=True)
```

### Composable architectures (`AI.ml.arch`)

Build models from swappable blocks — not canned themes. Every attention / FFN / norm / residual is independently constructible and registerable:

```python
AI.ml.arch.available()  # attention/ffn/norm/residual/embed/head/architecture

# Mix mechanisms per layer:
model = AI.ml.arch.compose(
    layers=[
        {"attention": "mha", "ffn": "mlp"},
        {"attention": "kda", "ffn": "moe", "ffn_options": {"n_experts": 8}},
        {"attention": "mla", "ffn": "latent_moe", "attention_options": {"gated": True}},
    ],
    n_embd=64, n_head=4, vocab_size=256, attn_res_every=2,
)

# Or build/register individual parts:
attn = AI.ml.arch.attention("delta", n_embd=64, n_head=4)
ffn = AI.ml.arch.ffn("swiglu", n_embd=64)
@AI.ml.arch.register_attention("mine")
def mine(**opts): ...

opt = AI.ml.optim.create("adamw", model.parameters(), lr=1e-3)
loss = AI.ml.loss.create("cross_entropy", label_smoothing=0.1)
```

Attention blocks: `mha` (KV cache), `linear`, `delta`, `gated_delta`, `kda`, `mla`. FFN: `mlp`, `swiglu`, `situ_mlp`, `moe`, `latent_moe`. See `examples/arch/main.py`.

Records carry features via `metadata["features"]` (explicit dict) → numeric metadata → text-derived fallback; the same convention feeds every backend. Native estimators (`simple:majority`, `simple:centroid`, `simple:knn`, `simple:linear_regression`) are real learners and persist as portable JSON. sklearn exposes `estimator.model` for `skops`/`joblib` persistence (aire never pickles); torch persists via `torch.save` with `weights_only=True` loads. The pandas bridge moves data both ways: `AI.ml.to_frame(dataset)` / `AI.ml.from_frame(df, target="label")`. Custom torch architectures plug in via `module_factory`. See `examples/ml/main.py` (runs offline).

## Tools

```python
from aire import AI
from aire.tools import SideEffect


@AI.tool(
    description="Search orders for a customer.",
    permissions=["database.read"],
    side_effect=SideEffect.READ_ONLY,
    timeout_seconds=5.0,
    retries=2,
)
async def search_orders(customer_id: str) -> list[dict]: ...
```

The decorator introspects the signature and docstring to produce a `ToolSpec`: name, description, JSON input/output schemas, permissions, timeout, retry policy, side-effect classification. Input validation is strict (`extra="forbid"`) — unknown arguments are rejected.

Builtin tools: `calculator` (restricted AST evaluator — no `eval`), `read_file`/`list_files` (sandboxed to a root), `http_get`, `current_time`.

## MCP (Model Context Protocol)

aire speaks MCP natively — zero dependencies, newline-delimited JSON-RPC 2.0 over stdio:

```bash
aire mcp-serve   # expose builtin + registered tools to any MCP host
```

```python
# Expose: any aire tool becomes an MCP tool
server = AI.mcp.server([search_orders])  # or AI.mcp.server() for builtins
await server.serve_stdio()

# Consume: any MCP server's tools become first-class aire Tools
async with await AI.mcp.connect(["python", "-m", "aire.mcp"]) as client:
    tools = await client.tools()  # remote schemas preserved
    result = await tools[0].execute({"expression": "2 ** 10"})
```

Remote tools keep their input schemas, so agents reason about them exactly like local ones — permissions, timeouts, retries and auditing included.

aire servers also expose **knowledge** so agents can learn the library through MCP itself — resources (`resources/list` / `resources/read`) and task prompts (`prompts/list` / `prompts/get`):

```python
async with await AI.mcp.connect(["aire", "mcp-serve"]) as client:
    guide = await client.read_resource("aire://guide")  # full usage guide
    manifest = await client.read_resource("aire://manifest")  # live AI.describe()
    plan = await client.get_prompt("aire_rag", {"docs": "./manuals"})
```

Built-in resources: `aire://guide`, `aire://manifest`, `aire://errors` (taxonomy + retryability), `aire://refs` (every `provider:name` scheme). Prompt templates: `aire_quickstart`, `aire_rag`, `aire_agent`, `aire_gateway`, `aire_ml`. See [docs/AIRE_FOR_AGENTS.md](docs/AIRE_FOR_AGENTS.md).

## Agents

```python
from aire import AI
from aire.agents import AgentConfig

agent = AI.agents.create_sync(
    "openai:gpt-4o-mini",
    tools=[search_orders],  # @AI.tool-decorated functions
    builtins=True,  # + calculator, read_file, list_files, http_get, ...
    config=AgentConfig(
        max_steps=12,
        token_budget=50_000,
        cost_budget_usd=0.25,
        permissions={"file.read"},
        approval_levels={"external_side_effect"},  # require approval for these
    ),
    memory="buffer",  # or jsonl:<path> for durable memory
)
result = agent.run_sync("What is 18% of 2450?")
print(result.output, result.status)  # completed | max_steps | budget_exceeded | failed
for step in result.steps:  # full audit trail
    print(step.index, step.kind, step.detail)
```

The executor is a deterministic state machine: **model call → permission check → (approval check) → tool execution → observation → completion decision**. Budgets are enforced by the `ExecutionContext` on every tick; state is resumable via `AgentState` checkpoints. Unknown tools and denied permissions become error observations fed back to the model — never crashes.

## Long-term memory

Buffer and JSONL memory remember the conversation; `AI.memory` remembers the *user*. It drops into any agent via the standard `memory=` parameter:

```python
memory = AI.memory.create(path=".aire/memory")  # episodic JSONL + semantic store
agent = AI.agents.create_sync("openai:gpt-4o-mini", memory=memory)

await memory.remember("The user prefers concise answers", salience=2.0)
hits = await memory.recall_semantic("answer style preference", k=3)

# Fold old episodes into durable facts with any model:
await memory.consolidate(model, max_facts=8)
```

Semantic recall is embedding-based, weighted by salience and a 30-day recency half-life. Consolidation distills old episodes into semantic facts and prunes the log — agents get better across runs without unbounded growth.

## Multi-agent teams

```python
researcher = AI.agents.create_sync("openai:gpt-4o-mini", name="researcher", tools=[...])
writer = AI.agents.create_sync("anthropic:claude-sonnet-4-5", name="writer")

# Every agent is also a tool:
tool = researcher.as_tool(description="Gather facts on a topic.")

# And a supervisor model can route subtasks across a team:
team = AI.agents.team({"researcher": researcher, "writer": writer}, supervisor="openai:gpt-4o-mini")
result = await team.run("Produce a market analysis report")
print(result.answer, result.delegations)  # auditable handoffs
```

The supervisor decides each round with validated structured output — delegate to one member or finish — and member outputs feed back as observations, so routing stays grounded in what specialists actually returned.

## Workflows

```python
from aire import AI

wf = AI.workflows.create("research_pipeline", max_visits=3)

wf.add("search", search_node)
wf.add("analyze", analysis_node, retries=2, timeout_seconds=30)
wf.add("verify", verification_node, requires_approval=True)
wf.add("publish", publish_node)

wf.connect("search", "analyze")
wf.connect("analyze", "verify")
wf.connect("analyze", "publish", when=lambda out: out["confidence"] > 0.9)  # conditional edge
wf.connect("verify", "publish")

result = await wf.run({"topic": "aire plugins"})  # parallel fan-out, fan-in joins
async for event in wf.run_stream({"topic": "x"}):  # live transition events
    print(event.kind, event.node)
```

Features: directed graphs, conditional branches (untaken branches marked `SKIPPED`), parallel execution of ready nodes, retries with backoff, per-node timeouts, **bounded loops** (cycles legal up to `max_visits`), JSON checkpoints after every node, resume from `WorkflowState`, human-approval nodes. Agents are valid node functions but are never required.

## Evaluation

```python
from aire import AI
from aire.evaluation.runner import save_report

report = AI.evaluate(
    agent,  # model, agent, knowledge pipeline, or callable
    "tests/evaluation.jsonl",  # or list[EvalCase] / list[dict]
    metrics=["accuracy", "groundedness", "contains", "latency", "cost"],
)  # sync; use AI.evaluator().run(...) for async
print(report.metric_summary())
save_report(report, "reports/run-42.json")
```

Built-in metrics: `exact_match`, `contains`, `regex_match`, `json_valid`, `semantic_overlap`, `groundedness`, `latency`, `cost`, `accuracy`, and `model_judge` (LLM-as-judge with any aire model). Every `CaseResult` preserves input, output, expected value, scores, error category, model, config, latency, usage, timestamp and trace id.

## Observability

```python
from aire import AI

tracer = AI.observe.tracer()  # OpenTelemetry-shaped spans
with tracer.span("pipeline.run", user_id="u1"):
    answer = await knowledge.ask("...")  # nested spans: retrieve, rerank, generate

metrics = AI.observe.metrics
metrics.snapshot()  # counters, gauges, latency histograms
```

Traces propagate through models, RAG, agents and deployment endpoints via `contextvars`; sensitive attribute keys (tokens, keys, secrets) are masked automatically. Exporters: in-memory (tests), JSONL (audit). The event bus broadcasts domain events (`model.generate`, `agent.tool_call`, …) for subscribers.

## Safety

```python
from aire import AI

chain = AI.safety.guardrails("pii", "injection", "secret")
verdicts = chain.check(user_input)  # raises SafetyError on blocking failures
clean = AI.safety.redact(user_input)  # remove PII + secrets

from aire.safety import ApprovalPolicy, SideEffect

policy = ApprovalPolicy(require_approval={SideEffect.EXTERNAL_SIDE_EFFECT, SideEffect.HIGH_IMPACT})
```

Every tool action carries a side-effect risk class: `read_only → reversible_write → external_side_effect → high_impact → prohibited`. High-impact actions require explicit approval unless a trusted policy grants it. Additional controls: sandboxed file access (path-traversal proof), `yaml.safe_load`-only deserialization (no pickle anywhere — enforced by test), secret redaction in traces, strict structured-output validation. See [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) for the threat model.

## Optimization

```python
from aire import AI

# Model routing by objective
router = AI.models.router(
    candidates=["openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5", "ollama:llama3.2"],
    objective="quality_under_budget",  # lowest_cost | lowest_latency | highest_quality | balanced
    cost_limit_usd=0.01,
)
text = await router.ask("...")  # routes + falls back + records history
decision = router.route(GenerationRequest.of("..."))
print(decision.chosen, decision.scores, decision.reason)

# Caching
cached = AI.models.cache(model)  # exact-match cache
semantic = SemanticCachedModel(model, embedder, threshold=0.95)  # similarity cache
```

## Multimodal, vision and audio

```python
from aire.core.content import AudioContent, ImageContent
from aire.multimodal import transcribe, describe_image
from aire.vision import VisionPipeline

text = await transcribe(asr_model, AudioContent.from_file("recording.mp3"))
caption = await describe_image(vision_model, ImageContent.from_file("photo.png"))

vision = VisionPipeline(vision_model)
label = await vision.classify("cat.png", labels=["cat", "dog"])
answer = await vision.vqa("chart.png", "What is the peak value?")
```

Conversions delegate to any model advertising the required capability (`SPEECH_RECOGNITION`, `VISION_INPUT`, …) — no vendor lock-in.

## Synthetic data

```python
from aire.synthetic import SyntheticGenerator

generator = SyntheticGenerator(model)
pairs = await generator.qa_pairs(document_text, n=20)  # grounded QA pairs
eval_dataset = await generator.augment(dataset, pairs_per_doc=3)  # dataset → QA eval set
```

Generation goes through structured output validation, so malformed samples are retried or dropped rather than silently polluting datasets.

## Training

```python
from aire.training import FunctionTrainer, TrainingConfig


async def step(epoch, dataset, config, state):
    # your framework code (torch, jax, numpy, ...) — fully framework-agnostic
    ...
    return {"loss": 0.42}, state  # (metrics, new_state)


trainer = FunctionTrainer(step, TrainingConfig(epochs=10, early_stopping_patience=2))
result = await trainer.fit(train_dataset)  # checkpoints + history in TrainResult
```

The trainer contract is framework-independent (checkpointing, early stopping, metrics, resume); PyTorch/TensorFlow/JAX arrive as lazy adapters (see [docs/ROADMAP.md](docs/ROADMAP.md)).

## Deployment

```python
from aire import AI

app = AI.deploy.api(
    agent_or_knowledge_or_model,
    title="support API",
    auth_token="...",  # optional bearer auth
    rate_limit_per_minute=60,  # optional per-client rate limit
    metrics=AI.observe.metrics,
)
# Endpoints: GET /health /ready /manifest /metrics
#            POST /v1/run (agent) | /v1/ask (knowledge) | /v1/generate (model)

artifacts = AI.deploy.artifacts(
    "./dist"
)  # Dockerfile, entrypoint.py, .env.template, requirements.lock
```

Errors surface as structured JSON (`error.code`, `error.context`) — the same contract as the Python API.

## Model gateway

aire doubles as a **model gateway**: one OpenAI-compatible server in front of every provider — local or hosted — with routing, fallbacks, auth, rate limits and observability built in. Point any existing OpenAI client at it unchanged.

```python
from aire import AI

app = AI.gateway.create(
    models=["ollama:llama3.2"],  # exposed under their own ref
    aliases={
        "smart": ["anthropic:claude-sonnet-4-5", "openai:gpt-4o-mini"],  # fallback chain
        "local": "lmstudio:qwen2.5-7b-instruct",
    },
    embeddings={"emb": "openai:text-embedding-3-small"},
    routing="first",  # or "round_robin" …
    # objective="lowest_cost",    # … or score candidates per request with ModelRouter
    auth_token="sk-internal",
    rate_limit_per_minute=600,
    metrics=AI.observe.metrics,
)
# Endpoints: POST /v1/chat/completions (streaming SSE supported)
#            POST /v1/messages           (Anthropic-compatible)
#            POST /v1/embeddings
#            GET  /v1/models  /v1/gateway/manifest  /health
```

Or from the shell (config falls back to the `gateway:` section of `aire.yaml`):

```bash
aire gateway -m ollama:llama3.2 \
             -a smart=anthropic:claude-sonnet-4-5,openai:gpt-4o-mini \
             --embed-alias emb=openai:text-embedding-3-small \
             --auth-token sk-internal --port 4000
```

Clients see a standard OpenAI API; responses additionally carry `aire.resolved_model` (and the `X-Aire-Resolved-Model` header) so routing decisions stay inspectable. Unknown models return OpenAI-shaped errors; a candidate that fails mid-chain falls back automatically.

Production guards are built in:

- **Circuit breakers** — a candidate that fails `failure_threshold` times in a row is skipped for `cooldown_seconds`, then half-open retried.
- **Cost budgets** — `budgets={"smart": 5.0}` caps daily spend per alias or ref; exhausted candidates are skipped, and when all are exhausted the gateway answers 429.
- **Anthropic-compatible endpoint** — existing Anthropic SDK clients can point at `POST /v1/messages` unchanged.
- **Request audit log** — `request_log="gateway.jsonl"` records model, resolved ref, tokens, cost and latency per request.

All of it configurable from `aire.yaml` under `gateway:` and visible in `GET /v1/gateway/manifest` (circuit states, budgets, today's spend).

## Configuration

Layered configuration with explicit priority — **Python arguments > project file > environment > user config > defaults**:

```yaml
# aire.yaml
project:
  name: support_agent

model:
  ref: openai:gpt-4o-mini
  temperature: 0.2

agent:
  max_steps: 12
  token_budget: 50000

safety:
  require_approval:
    - external_side_effect

gateway:
  aliases:
    smart: [anthropic:claude-sonnet-4-5, openai:gpt-4o-mini]
  embeddings:
    emb: local:hashing
  routing: first
  rate_limit_per_minute: 600
  budgets: {smart: 5.0}            # USD/day per alias or ref
  circuit_breaker: true
  failure_threshold: 3
  cooldown_seconds: 30
  request_log: logs/gateway.jsonl

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  groq:
    api_key: ${GROQ_API_KEY}
```

Environment overrides use `AIRE_<SECTION>__<KEY>` (e.g. `AIRE_MODEL__REF=ollama:llama3.2`). Secrets are `SecretStr` and excluded from serialization.

## CLI reference

```bash
aire init [NAME] [--dir DIR]      # scaffold a project (aire.yaml, app.py)
aire run "prompt" [--model REF]   # one-shot generation
aire evaluate DATASET [--model REF] [--metrics a,b] [--output report.json]
aire serve [--host H] [--port P] [--app-file app.py]
aire gateway [-m REF]... [-a public=ref[,ref...]]... [--objective O] [--port 4000]
aire mcp-serve                    # serve tools over MCP (stdio JSON-RPC)
aire inspect                      # show resolved config, registries, providers
aire plugins                      # list discovered plugins
aire doctor                       # environment/dependency/credential diagnostics
aire version
```

## Plugins and providers

Ship a provider without touching aire's core — implement the interface, register it, done:

```toml
# your package's pyproject.toml
[project.entry-points."aire.providers"]
myprovider = "my_package.aire_plugin:MyPlugin"
```

```python
def register(runtime):
    runtime.model_providers.register("myprovider", my_model_factory)
    return PluginInfo(name="myprovider", version="1.0.0", providers=["myprovider"])
```

Contract tests in `tests/contract` verify any model/embedder/vector-store against the shared interface. The complete contract — factory signatures, error rules, lifecycle, manifest requirements — is in [docs/PLUGIN_SPEC.md](docs/PLUGIN_SPEC.md).

**Included providers:** `openai`, `anthropic`, `huggingface`, `ollama`, `mock`/`echo` (offline), `callable` (any Python function), plus named OpenAI-compatible aliases — local: `lmstudio`, `llamacpp`, `llamafile`, `vllm`, `mlx`, `localai`, `tgi`; hosted: `groq`, `together`, `fireworks`, `deepseek`, `mistral`, `xai`, `openrouter`, `cerebras`, `perplexity` — and generic `openai_compatible` with a custom `base_url`. **Vector stores:** `local` (in-memory speed, JSON persistence), `sqlite` (embedded, transactional), `qdrant`, `chroma`, `pinecone`, `weaviate` (native BM25), `milvus`. **Graph stores:** `sqlite` (embedded triple store). All reached over plain HTTP through one shared client — no vendor SDKs.

## Examples

Runnable, offline, no credentials required:

| Example | What it demonstrates |
|---|---|
| [examples/rag_assistant](examples/rag_assistant/main.py) | The full vertical slice: ingest → index → ask with citations → evaluate |
| [examples/chatbot](examples/chatbot/main.py) | Tool-calling agent with budgets, permissions, step traces |
| [examples/model_router](examples/model_router/main.py) | Routing decisions with scores + cache hits |
| [examples/workflows](examples/workflows/main.py) | Branches, fan-in, streaming events, checkpoints |
| [examples/deployment_api](examples/deployment_api/main.py) | Auth-guarded FastAPI app + artifact generation |
| [examples/gateway](examples/gateway/main.py) | OpenAI-compatible gateway: aliases, fallback, streaming, embeddings |
| [examples/graphrag](examples/graphrag/main.py) | Knowledge graph ingestion + graph-grounded answers with citations |
| [examples/ml](examples/ml/main.py) | Estimator contract: native / sklearn / torch + pandas bridge |
| [examples/arch](examples/arch/main.py) | Compose attention/FFN blocks + optim/loss |
| [examples/teams](examples/teams/main.py) | Agent-as-tool, supervisor-routed team, long-term memory |

```bash
python examples/rag_assistant/main.py
```

## Documentation map

- [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) — product principles, subsystem scope, success criteria
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, module map, and 10 architectural decision records
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md) — the supported surface and stability guarantees
- [docs/PLUGIN_SPEC.md](docs/PLUGIN_SPEC.md) — writing providers and plugins
- [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) — threat model and the seven control classes
- [docs/ROADMAP.md](docs/ROADMAP.md) — 0.2 → 1.0 plan (training adapters, multimodal depth, multi-agent, stability)
- [CHANGELOG.md](CHANGELOG.md) — release history, updated with every major/minor version

## Development

```bash
git clone https://github.com/desenyon/aire.git && cd aire
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Quality gates (all must pass):
ruff check .
ruff format --check .
mypy src               # strict mode, 133 files
pytest                 # 340 tests: unit, contract, integration, security, performance
pytest tests/integration tests/security
```

Test suites: `tests/unit` (isolated), `tests/contract` (provider interface conformance), `tests/integration` (cross-module, including the offline vertical slice and mocked-provider HTTP), `tests/security` (injection, traversal, unsafe YAML, permission bypass), `tests/performance` (import time, throughput, latency budgets).

## License

Apache-2.0 — see [LICENSE](LICENSE).
