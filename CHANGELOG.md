# Changelog

All notable changes to aire are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): breaking changes bump the major
version (post-1.0), new subsystems and features bump the minor version, and
fixes/docs bump the patch version. **This file and the README are updated with
every major/minor release.**

## [0.3.3] — 2026-07-28

**0.3.2B hardening** — audit/fix/upgrade of partially built surfaces so
contracts match behavior (Wave 1) and integrations connect (Wave 2).

### Fixed / upgraded
- **PgVectorStore**: real ``vector`` + ``<=>`` when extension+dim available;
  honest ``jsonb_fallback`` + FTS keyword capability otherwise.
- **LoRATrainer.fit** (+ dry-run for CI); finetune recipe next-steps aligned.
- **GraphRAG query** injects community summaries into the prompt.
- **Skills**: ``apply`` / ``apply_skill`` binds tools + system prompt; recipes
  and ``AI.agents.create(skills=...)`` use it.
- **DurableSession** wired into ``Agent`` (persist steps/result on run).
- **Gateway stream** uses semantic cache lookup/store; ``describe()`` exposes
  hit/miss stats.
- **QuantMethod** narrowed to ``bitsandbytes|stub`` (no unwired literals).
- **Doctor --live**: skipped checks are ``ok: null`` (not false-green).
- **RedisCachedModel.clear()** SCAN+DEL under prefix.
- **DistillTrainer** + ``summarize_communities_async``; topology unit tests.

## [0.3.2] — 2026-07-28

Depth pass: evaluation/rerank, cost policies, quant/distill adapters, multimodal.

### Added
- **Eval metrics**: `faithfulness`, `embedding_similarity`, `bleu`, `rouge_l`;
  Evaluator accepts `embedder=`.
- **Rerankers**: `EmbeddingReranker`, `ModelReranker` (`cross_encoder` alias).
- **CostPolicy** for `ModelRouter` (per-request + daily budget, prefer-cheaper).
- **Quantizer** + **Distiller** training adapters (`AI.training.quantize` /
  `distill`); offline stub + pure-Python soft KL.
- **Vision** `detect()` + `ImageGenerationPipeline`; **Audio** `synthesize()` TTS.
- **Memory**: `resolve_memory("long-term")` / `long-term:<path>`.

## [0.3.1] — 2026-07-28

P0 correctness from the post-0.2 audit — memory/state, gateway timestamps,
OTLP drain, hybrid honesty.

### Fixed
- **Agent memory**: no duplicated user turns; `Agent.state` populated from the
  run; `reset()` clears memory; remove dead observation stub.
- **Gateway audit** `ts` is ISO-8601 UTC (not a date-only spend key).
- **OTLP**: `Runtime.aclose()` / `Tracer.flush()` drain pending batches.
- **Hybrid retrieval**: skip keyword fusion unless the store advertises
  `keyword-search` (hosted Pinecone/Chroma/Qdrant/Milvus stay vector-only);
  Pinecone text path respects metadata filters.

## [0.3.0] — 2026-07-28

Agent-operable AI platform: retrieval/ops depth, training hooks, multimodal
stubs, workers/schedulers, recipes, and ops UI — mostly offline-capable with
lazy optional extras.

### Added
- **GraphRAG communities**: offline label-propagation clustering + lexical
  summaries (`aire.graph.community`); lazy **Neo4j** GraphStore (`aire[neo4j]`).
- **HITL workflows**: `requires_approval` + `AI.workflows.hitl_node` /
  `NodeInteractiveApprover`.
- **Training**: PEFT/LoRA interface (`aire[peft]`), HPO random search + Optuna
  bridge (`aire[optuna]`), toy/torch LM trainer for `arch.compose` stacks.
- **Prompt optimization** eval-guided loop; **Redis** exact cache (`aire[redis]`);
  gateway **semantic cache**.
- **PDF** pipeline (`aire[pypdf]`), **voice agent** (ASR→agent→TTS stub),
  **video summarize** stub/pipeline.
- **Workers** (in-process + file queue), **scheduler** (interval; APScheduler
  optional), **OTel SDK bridge**, `aire doctor --live`, minimal **FastAPI UI**.
- **Project lock** (`aire.lock`), deepened **PolicyEngine**, **recipes**
  (`AI.recipe`), `aire scaffold`, facade namespaces: skills / schedule /
  workers / recipes / locks / topologies.

### Extras
- `pgvector`, `neo4j`, `redis`, `peft`, `optuna`, `pypdf` (also folded into
  `aire[all]`).

## [0.2.9] — 2026-07-28

Deep ML orchestration — expose the real sklearn / torch / keras surfaces under
one aire contract (compose, train, score, catalog), plus CatBoost and Polars.

### Added
- **Sklearn depth**: large estimator/transform catalogs (clustering, NB variants,
  GP, calibration, multi-output, encoders, text vectorizers, imputers, …);
  `partial_fit`, `sample_weight`, native `feature_importances_` / `coef_`.
- **Compose**: `ColumnTransformer`, `FeatureUnion` (`AI.ml.column_transformer`,
  `AI.ml.feature_union`).
- **Torch depth**: DataLoader, `validation_split`, AMP, `torch.compile`,
  gradient clipping, checkpoint restore-best, softmax probabilities, richer
  schedulers (`onecycle`, `cosine_warm`).
- **Keras depth**: full `compile(metrics=…)`, validation_split, class_weight,
  string/aire/keras callback mapping, predict probabilities.
- **Selection/metrics**: scoring registry (`roc_auc`, `log_loss`,
  `balanced_accuracy`, …), stratified CV, confusion matrix.
- **CatBoost** (`catboost:*`) + **Polars** bridge (`AI.ml.to_polars` /
  `from_polars`); `TaskType.CLUSTERING` / `MULTI_LABEL`.
- Extras: `aire[catboost]`, `aire[polars]`; `aire[all]` now includes torch.

## [0.2.8] — 2026-07-28

Finish the ML orchestrator: one Estimator / Transform / Pipeline contract across
native, scikit-learn, PyTorch, Keras, XGBoost, and LightGBM — with training
callbacks, model selection direction, and a unified train facade.

### Added
- **`Transform` + `Pipeline`**: chain `native:*` / `sklearn:*` preprocessors into
  a final estimator under one fit/predict API (`AI.ml.pipeline`, `AI.ml.transform`).
- **Backends**: `keras:mlp`, `xgboost:classifier|regressor`,
  `lightgbm:classifier|regressor` (lazy extras `aire[keras|xgboost|lightgbm]`).
- **Torch training depth**: wire `AI.ml.optim` / `AI.ml.loss`, batching,
  schedulers (`step` / `cosine` / `plateau`), callbacks (`EarlyStopping`,
  `HistoryCallback`).
- **Expanded sklearn zoo** + sklearn transformer adapters (PCA, imputers,
  selectors, scalers, …).
- **`AI.ml.random_search`**, `direction=` on grid/random search, `AI.ml.train`
  (optional transform pipeline), richer `catalog` / `backends` / `describe`.

### Changed
- Estimator registration always covers every backend (idempotent factory).

## [0.2.7] — 2026-07-28

Composable neural architecture blocks — assemble any stack from parts, plus
first-class optimizers and losses. Not model themes: every attention / FFN /
norm / residual is independently constructible, swappable, and registerable.

### Added
- **`aire.ml.arch` / `AI.ml.arch`**: block registries for
  `attention` (mha, linear, delta, gated_delta, kda, mla),
  `ffn` (mlp, swiglu, situ_mlp, moe, latent_moe),
  `norm` (layernorm, rmsnorm, identity),
  `residual` (add, attn_res), `embed`, `head`.
- **Compose API**: `AI.ml.arch.attention/ffn/norm/residual/block(...)`,
  `AI.ml.arch.compose(layers=[...])` with per-layer overrides,
  `register_attention` / `register_ffn` / `register_architecture` for
  user-defined parts and full stacks. Optional thin recipes
  (`uniform_mha`, `hybrid_cycle`, …) are compositions, not themes.
- **`AI.ml.optim`**: sgd, adam, adamw, rmsprop, adagrad.
- **`AI.ml.loss`**: cross_entropy, nll, mse, l1, huber, smooth_l1, bce,
  kl_div, cosine, ctc, moe_load_balance.
- Example `examples/arch/main.py`; tests in `tests/unit/test_ml_arch.py`.

## [0.2.6] — 2026-07-28

Deepen existing ML orchestration and gateway surfaces — richer metrics,
model selection, and operator observability.

### Added
- **ML evaluation depth**: full classification reports (per-class + macro/micro
  precision/recall/F1), regression R² alongside MAE/RMSE;
  `Estimator.cross_validate` / `AI.ml.cross_validate`; exhaustive
  `AI.ml.grid_search` with inner k-fold CV; model-agnostic
  `Estimator.feature_importance` (permutation); `AI.ml.catalog()` catalog.
- **Gateway operator APIs**: `GET /v1/health` (rich payload),
  `GET /v1/gateway/spend` (today's spend, budgets, remaining), response
  headers `X-Aire-Cost-Usd` / `X-Aire-Input-Tokens` / `X-Aire-Output-Tokens`
  on chat and Anthropic messages; `Gateway.spend_today()`.

## [0.2.5] — 2026-07-27

Final sweep of the 0.2.x hardening series: documentation alignment and a last
correctness review across the remaining modules.

### Changed
- Packaged agent guide (`aire/mcp/guide.md`) covers the new loaders and ML
  subsystem; `docs/AIRE_FOR_AGENTS.md` cross-links the doc map.
- `docs/PUBLIC_API.md`: new "Model creation (ML)" section; MCP knowledge
  methods, approval policies, OTLP exporter, trainer/workflow resume APIs,
  and loader coverage documented (workflow checkpoint API corrected).
- `docs/ROADMAP.md`: 0.2.1–0.2.5 entries recorded; README dev stats updated
  (326 tests, 116 strict-mypy files).
- Reviewed `models/retry.py`, `rag/retriever.py`, `rag/rerank.py`,
  `knowledge_assistant.py`, `training/trainer.py` — verified correct; no
  changes required.

## [0.2.4] — 2026-07-27

Hardening pass 2: workflow checkpoint resume actually works now, approval
policies ship in the box, and the data loaders cover HTML and tabular files.

### Added
- **Approval policies** (`src/aire/agents/approvals.py`): `RuleApprover`
  (auto-approve below a side-effect severity, per-tool allow/deny overrides,
  built-in audit trail) and `InteractiveApprover` (human-in-the-loop stdin
  prompts with per-session `always`/`never` memory, non-blocking).
  `AI.agents.approver("rule" | "interactive")` facade helper.
- **HTML loader**: `html_to_text()` (stdlib-only: strips script/style/tags,
  decodes entities, preserves paragraph breaks); `.html`/`.htm` files,
  HTML pages in directory loads, and HTML URLs all produce clean text.
- **Tabular loaders**: `.parquet`/`.xlsx`/`.xls` via lazy pandas
  (`pip install aire[ml]`), rows → records through the standard path.

### Fixed
- **Workflow resume was broken** for any multi-node graph: edge firing counts
  are runtime-local, so resumed runs never scheduled downstream nodes of
  completed predecessors; the persisted error carried into the new run; and
  FAILED nodes were never eligible for re-execution. `Workflow.resume()` now
  reconstructs edge firings from persisted statuses + outputs, clears the
  failure, retries failed nodes, and seeds the first wave when the entry
  node is already terminal. Untaken conditional branches stay skipped.
- Added `Workflow.load_checkpoint(path)` + `Workflow.resume(path?)`.
- 12 new tests (`tests/unit/test_hardening_024.py`).

## [0.2.3] — 2026-07-27

Hardening pass 1: correctness fixes in the caches, resumable training, and a
real OpenTelemetry exporter.

### Added
- **OTLP exporter** (`src/aire/observability/otlp.py`): batched OTLP/HTTP+JSON
  spans to any collector (`/v1/traces`) over httpx — no OTel SDK required.
  Failures are counted, never raised; configurable via
  `observability.exporter: otlp` + `otlp_endpoint` in `aire.yaml`.
- **Resumable training**: `FunctionTrainer.fit(dataset, resume_from=...)`
  continues from a checkpoint (epoch, step state, best metric), and
  `FunctionTrainer.load_checkpoint(path)` loads one explicitly.

### Fixed
- **Cache poisoning via mutation** (`CachedModel`, `SemanticCachedModel`):
  cached `GenerationResult`s were returned by reference, so a caller mutating
  a result corrupted every future hit. Both stores and hits now use deep
  copies.
- **Semantic cache ignored generation parameters**: a request differing only
  in `temperature`/`response_format`/etc. could be served a stale entry.
  Hits now require an exact parameter signature in addition to prompt
  similarity — structured-output requests can never be served plain-text
  entries.
- **Tracer `mask_fields` case sensitivity**: fields are now matched
  case-insensitively (`API_Key` masks `api_key`), matching the documented
  behavior.
- 10 new tests (`tests/unit/test_hardening_023.py`).

## [0.2.2] — 2026-07-27

MCP knowledge: agents can now learn how to operate aire through MCP itself —
resources (docs/manifests) and prompts (task templates) ride alongside tools.

### Added
- **MCP resources** (`src/aire/mcp/knowledge.py`): `resources/list` +
  `resources/read` for `aire://guide` (full usage guide, packaged with the
  library at `aire/mcp/guide.md`), `aire://manifest` (live `AI.describe()`),
  `aire://errors` (error taxonomy with retryability) and `aire://refs`
  (every `provider:name` scheme).
- **MCP prompts**: `prompts/list` + `prompts/get` with five task templates —
  `aire_quickstart`, `aire_rag`, `aire_agent`, `aire_gateway`, `aire_ml` —
  with declared arguments, defaults and safe placeholder substitution.
- **Client support**: `MCPClient.list_resources()`, `read_resource(uri)`,
  `list_prompts()`, `get_prompt(name, arguments)`.
- Server `knowledge=True` flag (disable to expose tools only); capabilities
  advertised in `initialize`; `MCPServer.describe()` lists resources/prompts.
- `docs/AIRE_FOR_AGENTS.md`: agent entry point tying MCP knowledge to the
  repo documentation map.
- 13 new tests (`tests/unit/test_mcp_knowledge.py`) including a full stdio
  round-trip reading the guide and rendering prompts through a real client.

## [0.2.1] — 2026-07-27

Model creation: aire now orchestrates the ML ecosystem — train, evaluate and
persist ML models through one contract, offline by default.

### Added
- **`aire.ml` — the `Estimator` contract** (`src/aire/ml/estimator.py`):
  `fit(dataset, target=)` → `predict(records)` → `evaluate` → `save`/`load`,
  async end to end with blocking compute in worker threads. Shared feature
  convention: explicit `record.metadata["features"]` dicts → numeric metadata
  → text-derived fallback.
- **Native estimators** (`src/aire/ml/native.py`, refs `simple:*`): real
  zero-dependency learners — `majority` (baseline w/ probabilities),
  `centroid` (nearest class centroid, z-normalized), `knn`, and
  `linear_regression` (gradient descent). JSON-portable persistence.
- **scikit-learn backend** (`src/aire/ml/sklearn_adapter.py`, refs
  `sklearn:*`): 13 named estimators (`random_forest`, `logistic_regression`,
  `gradient_boosting`, `svm`, `mlp`, ...) plus any dotted class path;
  `predict_proba` support. Fitted-model persistence is delegated to the
  caller via `skops.io`/`joblib` on `estimator.model` — aire never pickles
  (enforced by the security test-suite). Lazy — requires `aire[ml]`.
- **PyTorch backend** (`src/aire/ml/torch_adapter.py`, refs `torch:*`):
  configurable MLP trainer for classification/regression with
  `module_factory` for custom `nn.Module`s; `torch.save` persistence with
  `weights_only=True` loads (tensors + primitives, no executable pickle).
  Lazy — requires `aire[torch]`.
- **pandas bridge** (`src/aire/ml/pandas_bridge.py`): `frame_to_dataset`,
  `dataset_to_frame`, `predictions_to_frame`, `available_backends()`.
  Lazy — requires `aire[ml]`.
- **Facade**: `AI.ml.create(spec, **options)`, `AI.ml.fit(spec, dataset)`,
  `AI.ml.backends()`, `AI.ml.to_frame/from_frame`, `AI.ml.describe()`.
- `Runtime.estimators` registry property; `ml` and `torch` extras in
  `pyproject.toml`.
- 17 new tests (`tests/unit/test_ml.py`) and `examples/ml/main.py`
  (fully offline).

## [0.2.0] — 2026-07-27

Knowledge graphs, MCP, long-term memory, embedded stores, multi-agent teams
and gateway hardening: six new subsystems, all offline-capable, all following
the same registry/ref/manifest contracts.

### Added

- **Knowledge graphs and GraphRAG** (`aire.graph`, `AI.graph`): documents →
  triples → graph-grounded answers with citations.
  - `KnowledgeGraph` pipeline: chunk → extract → store → fused query (graph
    neighborhood + vector retrieval) → grounded `Answer` with citations.
  - Two extractors: `ModelGraphExtractor` (typed triples via any model's
    structured output) and `LexicalGraphExtractor` (deterministic, zero-model,
    offline: capitalized phrases + sentence co-occurrence).
  - `SQLiteGraphStore` (`sqlite:<path>` / `sqlite:memory`): embedded,
    transactional triple store on stdlib `sqlite3` — BFS neighborhoods, entity
    matching, merge-on-ingest. New `Runtime.graph_stores` registry kind.
- **Model Context Protocol** (`aire.mcp`, `AI.mcp`): zero-dependency
  newline-delimited JSON-RPC 2.0 over stdio (protocol `2025-06-18`).
  - `MCPServer` exposes any aire tools (`initialize`, `ping`, `tools/list`,
    `tools/call`); `aire mcp-serve` and `python -m aire.mcp` serve builtin +
    registered tools to any MCP host.
  - `MCPClient` spawns any MCP server subprocess and adapts every remote tool
    into a first-class aire `Tool` (remote input schemas preserved).
- **Long-term agent memory** (`aire.memory`, `AI.memory`): `LongTermMemory`
  implements the agent `Memory` interface (drop-in for `Agent(memory=...)`)
  and adds `remember()` / `recall_semantic()` (embedding recall weighted by
  salience and 30-day recency half-life), episodic JSONL persistence, and
  `consolidate(model)` — folding old episodes into durable semantic facts.
- **Embedded SQLite vector store** (`sqlite:<path>`, `aire.rag.sqlite`):
  same BM25 + cosine semantics as the local store with write-through
  transactional persistence. Registered alongside `local:` on every runtime.
- **Hosted vector store adapters**: `pinecone:<index>`, `weaviate:<class>`
  (native server-side BM25) and `milvus:<collection>` — pure-httpx REST,
  consistent with the existing qdrant/chroma adapters.
- **Multi-agent teams** (`aire.agents.team`, `AI.agents.team`):
  `agent.as_tool()` wraps any agent as a `task: str → str` Tool; `Team` runs a
  supervisor model that routes subtasks to specialist members via validated
  structured decisions, feeds observations back, and synthesizes the answer.
- **Gateway hardening**:
  - Per-candidate circuit breakers (`failure_threshold`, `cooldown_seconds`):
    failing refs are skipped while open and half-open after cooldown.
  - Daily cost budgets (`budgets={"alias": usd}`): over-budget candidates are
    skipped; all-exhausted returns 429.
  - Anthropic-compatible `POST /v1/messages` endpoint (system prompts, content
    blocks, `stop_reason` mapping).
  - JSONL request audit log (`request_log="path.jsonl"`).
  - Unstructured provider exceptions are now wrapped in `ProviderError` so the
    gateway always answers with a structured error body.
  - All of the above configurable from `aire.yaml` (`gateway.budgets`,
    `gateway.circuit_breaker`, `gateway.request_log`, ...).
- Facade namespaces `AI.graph`, `AI.memory`, `AI.mcp`, plus `AI.agents.team()`;
  `aire mcp-serve` CLI command; top-level exports `KnowledgeGraph`,
  `LongTermMemory`, `MemoryEntry`, `Team`, `TeamResult`.

### Fixed

- **Vector store persistence dropped embeddings**: `LocalVectorStore.save()`
  serialized chunks via `model_dump`, which excludes embeddings — reloaded
  stores were unsearchable by vector. Embeddings are now persisted.

[0.2.0]: https://github.com/naitikgupta/aire/compare/v0.1.1...v0.2.0

## [0.1.1] — 2026-07-27

Model routing and gateway release: aire now works with every local model server
and every OpenAI-compatible API through named refs, and can itself act as an
OpenAI-compatible gateway in front of all of them.

### Added

- **Model gateway** (`aire.deployment.gateway`, `AI.gateway`, `aire gateway`):
  an OpenAI-compatible FastAPI server exposing `POST /v1/chat/completions`
  (with streaming SSE), `POST /v1/embeddings`, `GET /v1/models`,
  `GET /v1/gateway/manifest` and `/health` in front of any aire model refs.
  - Routing modes: ordered fallback chains (`routing="first"`), `round_robin`,
    or per-request objective scoring via `ModelRouter` (`objective="lowest_cost"`,
    `"highest_quality"`, ...).
  - Bearer auth, per-client rate limiting, tracing spans and per-model
    metrics; OpenAI-shaped error responses with `aire.resolved_model` on every
    answer plus the `X-Aire-Resolved-Model` header.
  - Request `model` accepts exposed aliases or any `provider:name` ref directly.
- **Universal OpenAI-compatible provider** (`aire.integrations.openai_compat`):
  named endpoint aliases with default base URLs and env vars, registered lazily:
  - Local servers (no key required): `lmstudio`, `llamacpp`, `llamafile`,
    `vllm`, `mlx`, `localai`, `tgi`.
  - Hosted APIs: `groq`, `together`, `fireworks`, `deepseek`, `mistral`, `xai`,
    `openrouter`, `cerebras`, `perplexity`.
  - Generic `openai_compatible:<model>` with explicit `base_url`/`api_key` for
    any other compatible server.
  - Machine-readable catalog via `AI.gateway.endpoints()`.
- **Configuration**: `gateway:` section in `aire.yaml` (`GatewayConfig`) —
  models, aliases, embeddings, routing, objective, auth token, rate limit;
  `AI.gateway.create()` falls back to it.
- **CLI**: `aire gateway` command (`-m/--model`, `-a/--alias`, `--embed-alias`,
  `--routing`, `--objective`, `--auth-token`, `--rate-limit`, `--port`).
- **Docs**: README "The problems aire fixes" section, "Local models and
  OpenAI-compatible endpoints" and "Model gateway" guides; `examples/gateway`.

### Changed

- Lazy provider hinting (`_maybe_hint_integration`) now maps provider prefixes
  to integration modules explicitly and covers the whole OpenAI-compatible
  alias catalog — touching any single alias registers all of them.

## [0.1.0] — 2026-07-26

Initial public release.

### Added

- **Core runtime**: layered config (`aire.yaml` / env / user config), DI
  `Runtime`, thread-safe registries, plugin discovery via entry points, event
  bus, resource lifecycle, safe serialization, structured `AireError`
  hierarchy with stable codes and `retryable` flags.
- **Models**: universal async `Model` / `EmbeddingModel` interfaces, normalized
  `ModelInfo`/`Usage` metadata, `provider:name` resolution with instance
  caching, retries with backoff, offline builtins (`mock:echo`,
  `local:hashing`, `callable:<fn>`).
- **Providers**: `openai`, `anthropic`, `ollama`, `huggingface`; vector stores
  `local`, `qdrant`, `chroma`.
- **Data**: loaders (jsonl/json/csv/text/directory/url/memory), chainable
  immutable `Dataset` ops (validate/dedupe/filter/split/sample), chunkers
  (fixed/sentence/recursive/semantic).
- **RAG**: `Knowledge` pipeline with hybrid retrieval, reranking, cited
  answers, persistent local vector store.
- **Tools & agents**: `@AI.tool` decorator with JSON-schema contracts,
  permissions and side-effect classification; deterministic agent state
  machine with step/token/cost budgets and approval policies.
- **Workflows**: graph engine with conditional branches, parallel fan-out,
  retries, checkpoints and streaming events.
- **Evaluation & observability**: metrics and LLM judges, reports;
  OpenTelemetry-shaped tracing, cost/latency/token metrics, event history.
- **Safety & optimization**: PII/secret/injection guardrails, redaction;
  exact/semantic caching, objective-based `ModelRouter`.
- **Deployment & CLI**: FastAPI app factory with auth/rate limits/manifest,
  Dockerfile + artifact generation; `aire` CLI (init/run/evaluate/serve/
  inspect/plugins/doctor/version).
- **Docs & tests**: PRODUCT_SPEC, ARCHITECTURE, PUBLIC_API, PLUGIN_SPEC,
  SECURITY_MODEL, ROADMAP; five runnable offline examples; 217 tests across
  unit/contract/integration/security/performance suites.

[0.1.1]: https://github.com/desenyon/aire/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/desenyon/aire/releases/tag/v0.1.0
