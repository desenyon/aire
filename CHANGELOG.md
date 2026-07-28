# Changelog

All notable changes to aire are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): breaking changes bump the major
version (post-1.0), new subsystems and features bump the minor version, and
fixes/docs bump the patch version. **This file and the README are updated with
every major/minor release.**

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
