# aire — Agent Usage Guide

This document teaches an agent (or human) how to use aire end to end. Every
component follows the same contracts: **`provider:name` refs**, **registries**,
**`.describe()` manifests**, **structured `AireError`s**, and **offline-first
defaults**. When unsure, call `.describe()` on the object or `AI.describe()`.

## 1. The facade

```python
from aire import AI

AI.describe()  # full library manifest: version, namespaces, registries
AI.ml.backends()  # which ML backends are importable
```

Namespaces: `models`, `data`, `rag`, `graph`, `memory`, `ml`, `mcp`, `agents`,
`workflows`, `training`, `observe`, `safety`, `deploy`, `gateway`, `tool`,
`tools()`. All namespaces operate on the default runtime; `AI.configure(...)`
rebuilds it from `Settings`.

Sync/async: every async API has a `*_sync` variant or use `aire.core.async_utils.run_sync`.
Never call a sync wrapper inside a running event loop.

## 2. Models — using them

```python
model = AI.models.get("openai:gpt-4o-mini")  # any provider:name ref
text = await model.ask("Explain RAG in one sentence.")
data = await model.ask_structured(prompt, MyPydanticModel)  # validated structured output
result = await model.generate(GenerationRequest(...))  # full control
```

Provider families (all `provider:name`): `openai`, `anthropic`, `ollama`,
`huggingface`, `mock`, `echo`, plus every OpenAI-compatible endpoint:
`lmstudio`, `llamacpp`, `vllm`, `mlx`, `groq`, `together`, `mistral`,
`openrouter`, `deepseek`, `xai`, `fireworks`, `cerebras`, `openai_compatible:base_url=...`.
Missing API keys/deps raise `ConfigurationError` with an actionable hint.

Routing & caching:

```python
routed = AI.models.route(["groq:llama-3.3-70b", "openai:gpt-4o-mini"], objective="cost")
cached = AI.models.cached(model)  # exact + semantic response cache
```

## 3. Data

```python
ds = AI.data.load("./docs")  # dir, .jsonl/.json/.csv/.html/.parquet/.xlsx, URL, or list
chunks = AI.data.chunk(ds, strategy="recursive", size=512)
```

HTML files/URLs are converted to clean text automatically; parquet/excel need
`aire[ml]` (pandas).

`Dataset` iterates `Record(text, metadata)`; `record.metadata` carries
everything downstream (chunk offsets, labels, features).

## 4. RAG

```python
rag = AI.rag.create(embedder="builtin:hash", store="local:default")  # offline
report = await rag.ingest("./documents")
answer = await rag.query("How do refunds work?")  # Answer(text, citations)
```

Stores: `local:default` (in-memory, `.save()` JSON), `sqlite:<path>` (embedded,
durable), `qdrant:*`, `chroma:*`, `pinecone:*`, `weaviate:*`, `milvus:*`.
Pipeline: chunker → embedder → hybrid retriever (vector + BM25, RRF) →
reranker → grounded prompt with numbered citations.

## 5. Knowledge graphs (GraphRAG)

```python
graph = AI.graph.create()  # sqlite triple store + lexical extractor
graph = AI.graph.create(model="ollama:llama3.2")  # model-driven typed triples
await graph.ingest("./documents")
facts = await graph.subgraph("question terms")  # Subgraph.as_context()
answer = await graph.query("How do refunds relate to chargebacks?")
```

## 6. Memory

```python
mem = AI.memory.create(path="memory.jsonl")  # episodic + semantic, persistent
await mem.remember("User prefers terse answers", kind="semantic")
facts = await mem.recall_semantic("communication style")
agent = AI.agents.create_sync(model, memory=mem)
await mem.consolidate(model)  # summarize old episodes into semantic facts
```

## 7. Tools and agents

```python
from aire import tool


@tool(description="Add two numbers", side_effect="none")
def add(a: int, b: int) -> int:
    return a + b


agent = AI.agents.create_sync("openai:gpt-4o-mini", tools=[add], max_steps=8)
result = await agent.run("What is 40+2?")
result.output, result.steps, result.usage
```

Multi-agent:

```python
researcher_tool = researcher.as_tool()  # agent → Tool
team = AI.agents.team(supervisor="openai:gpt-4o", members=[researcher, writer])
result = await team.run("Write a market brief on EV batteries")
```

## 8. Model creation (ML)

```python
est = await AI.ml.fit("simple:centroid", dataset)  # offline native
est = await AI.ml.fit("sklearn:random_forest", dataset)  # needs aire[ml]
est = await AI.ml.fit("torch:mlp", dataset, hidden=(64, 32), optimizer="adamw")
est = await AI.ml.fit("keras:mlp", dataset)  # aire[keras]
est = await AI.ml.fit("xgboost:classifier", dataset)  # aire[xgboost]
pipe = AI.ml.pipeline([("scale", "native:standard_scaler"), ("clf", "simple:centroid")])
await pipe.fit(dataset)
await est.evaluate(dataset)
await est.predict(records)
est.save("model.json")
AI.ml.create("simple:centroid").load("model.json")
```

Feature convention: `record.metadata["features"]` dict → numeric metadata →
text-derived fallback. Targets come from `record.metadata[target]`.
pandas bridge: `AI.ml.to_frame(ds)` / `AI.ml.from_frame(df, target="label")`.
aire never pickles: sklearn/xgboost/lightgbm persist via their own APIs on
`est.model`; torch uses `torch.save` with `weights_only=True`.
`AI.ml.catalog()`, `AI.ml.cross_validate(...)`, `AI.ml.grid_search(...)`,
`AI.ml.random_search(...)`, `est.feature_importance(...)` for model selection.

### Composable neural blocks

```python
model = AI.ml.arch.compose(
    layers=[
        {"attention": "mha", "ffn": "mlp"},
        {"attention": "kda", "ffn": "moe", "ffn_options": {"n_experts": 8}},
    ],
    n_embd=64, n_head=4,
)
AI.ml.arch.available()          # all registered block kinds
AI.ml.arch.attention("mla", n_embd=64, n_head=4, gated=True)
AI.ml.optim.create("adamw", model.parameters(), lr=1e-3)
AI.ml.loss.create("cross_entropy")
```

Register your own: `@AI.ml.arch.register_attention("mine")` /
`register_ffn` / `register_architecture`.

## 9. Gateway (serve models as APIs)

```python
app = AI.gateway.create(routes={"fast": ["groq:llama-3.3-70b", "ollama:llama3.2"]})
# OpenAI:  POST /v1/chat/completions, /v1/embeddings, GET /v1/models
# Anthropic: POST /v1/messages ; manifest: GET /v1/gateway/manifest
```

Or `aire gateway --port 8080`. Config in `aire.yaml`: budgets (USD/day per
alias), circuit breakers, rate limits, request logging (JSONL).
Health: `GET /health` and `GET /v1/health`. Spend: `GET /v1/gateway/spend`.
Responses include `X-Aire-Resolved-Model`, `X-Aire-Cost-Usd`, token headers.

## 10. Evaluation

```python
report = await AI.evaluate(model, cases, metrics=["accuracy", "groundedness"])
```

Cases: `EvalCase(input, expected, context, metadata)`. Built-in metrics:
`exact_match`, `accuracy`, `contains`, `semantic_overlap`, `json_valid`,
`regex_match`, `groundedness`, `latency`, `cost`, `model_judge`.

## 11. MCP (this protocol)

Server: `aire mcp-serve` or `AI.mcp.server([tools])`. Client:

```python
client = await AI.mcp.connect(["aire", "mcp-serve"])
tools = await client.tools()  # remote tools as aire Tools
docs = await client.read_resource("aire://guide")
prompt = await client.get_prompt("aire_rag")
```

## 12. Errors

All failures are `AireError` subclasses with `.code`, `.context`,
`.retryable`: `ConfigurationError` (bad setup — fix config),
`AuthenticationError`, `RateLimitError` (retryable), `ProviderError`
(retryable), `TimeoutError` (retryable), `NotFoundError` (unknown ref/name),
`PermissionDeniedError`, `BudgetExceededError`, `DataError`, `PluginError`,
`ValidationError`, `MCPError`. Catch `AireError`, inspect `code`, honor
`retryable`.

## 13. Configuration

`aire.yaml` (or env `AIRE_` prefixed): providers, models.default, gateway
(host/port/routes/budgets/circuit breaker), telemetry. `AI.configure()`
reloads. Secrets only via env vars — never in config files.

## 14. Rules of thumb for agents

1. Discover, don't assume: `.describe()` everything; `AI.models.list()`,
   `AI.ml.backends()`, registry `.names()`.
2. Prefer offline defaults in tests: `mock:` models, `builtin:hash` embedders,
   `simple:*` estimators, `LexicalGraphExtractor`.
3. Never pickle; never call sync wrappers inside an event loop.
4. Budgets and permissions are enforced by the runtime — don't work around them.
5. On error: read `exc.code` and `exc.context`; retry only if `exc.retryable`.
