# aire — Roadmap

## 0.2.7 (current) — composable neural architecture blocks

- [x] First-class attention / FFN / norm / residual / embed / head blocks with
      register-your-own factories; compose arbitrary per-layer stacks
- [x] Optimizers + loss functions as aire registries (`AI.ml.optim` / `AI.ml.loss`)

## 0.2.6 — deepen ML + gateway

- [x] Richer ML metrics (classification report, R²), k-fold CV, grid search,
      permutation importance, `AI.ml.catalog()`
- [x] Gateway `/v1/health`, `/v1/gateway/spend`, cost/token response headers

## 0.2.1–0.2.5 — model creation, agent knowledge & hardening

- [x] `aire.ml` (0.2.1): the `Estimator` contract — `fit(dataset) → predict →
      evaluate → save/load`; native `simple:*` estimators (offline), sklearn
      (`sklearn:*`) and torch (`torch:*`) adapters, pandas bridge, `AI.ml`
      facade; aire never pickles (torch loads use `weights_only=True`)
- [x] MCP knowledge (0.2.2): `aire://guide|manifest|errors|refs` resources +
      task prompts (`aire_rag`, `aire_agent`, `aire_gateway`, `aire_ml`,
      `aire_quickstart`) — agents learn aire through MCP itself;
      `docs/AIRE_FOR_AGENTS.md`
- [x] Hardening 1 (0.2.3): cache mutation/parameter-signature fixes, OTLP
      exporter (batched, failure-safe), resumable `FunctionTrainer`
- [x] Hardening 2 (0.2.4): working workflow checkpoint resume (edge-firing
      reconstruction, failed-node retry), `RuleApprover`/`InteractiveApprover`,
      HTML + parquet/excel loaders
- [x] Sweep (0.2.5): docs polish, PUBLIC_API/roadmap alignment, 326 tests green

## 0.2 — knowledge, memory & multi-agent

Six new subsystems, all offline-capable: knowledge graphs, MCP, long-term
memory, embedded stores, multi-agent teams, and gateway hardening.

- [x] `aire.graph`: GraphRAG pipeline (extract → triple store → graph-grounded
      answers with citations), lexical + model-driven extractors
- [x] Embedded `SQLiteGraphStore` (`sqlite:<path>`), `Runtime.graph_stores`
- [x] `aire.mcp`: MCP stdio server (`aire mcp-serve`) + client adapting remote
      MCP tools into aire Tools — zero dependencies
- [x] `aire.memory`: long-term episodic + semantic memory, salience/recency
      recall, model-driven consolidation, agent drop-in
- [x] Embedded `sqlite:` vector store; hosted `pinecone:`, `weaviate:` (native
      BM25), `milvus:` REST adapters
- [x] Multi-agent: `agent.as_tool()`, supervisor-routed `Team` with auditable
      handoffs (`AI.agents.team`)
- [x] Gateway hardening: circuit breakers, daily cost budgets, Anthropic
      `/v1/messages`, JSONL request audit log
- [x] Fix: vector store persistence no longer drops embeddings on save

## 0.1.1 — model routing & gateway

Every local model server and every OpenAI-compatible API reachable through
named refs; aire can itself serve as the OpenAI-compatible gateway in front of
all of them.

- [x] Model gateway: OpenAI-compatible `/v1/chat/completions` (streaming SSE),
      `/v1/embeddings`, `/v1/models`, manifest endpoint
- [x] Gateway routing: ordered fallback chains, round-robin, objective-based
      scoring via ModelRouter; auth, rate limits, tracing, metrics
- [x] Local model refs: llamacpp, lmstudio, vllm, mlx, localai, llamafile, tgi
- [x] Hosted OpenAI-compatible refs: groq, together, fireworks, deepseek,
      mistral, xai, openrouter, cerebras, perplexity
- [x] Generic `openai_compatible:` provider with custom base_url/api_key
- [x] `gateway:` config section, `AI.gateway` facade, `aire gateway` CLI
- [x] CHANGELOG.md — updated with every major/minor release from now on

## 0.1 — architecture proof

The vertical slice release: load → chunk → embed → store → retrieve → answer
with citations → evaluate → trace → deploy, all offline-capable.

- [x] Core runtime (config, registries, plugins, events, context, lifecycle)
- [x] Unified model + embedding interfaces, provider registry, retries
- [x] Providers: openai, anthropic, ollama, huggingface, mock/echo
- [x] Vector stores: local (persistent), qdrant, chroma
- [x] Data: loaders (files/dirs/URLs/memory), dataset ops, chunkers
- [x] RAG: hybrid retrieval (vector + keyword, RRF), reranking, citations
- [x] Tools: decorator, registry, permissions, builtin tools, sandboxing
- [x] Agents: deterministic state machine, memory, budgets, approval
- [x] Workflows: graph engine, branches, parallel, retries, bounded loops,
      checkpoints, streaming events
- [x] Evaluation: metric registry, model judges, reports
- [x] Observability: tracing (OTel-shaped), metrics, event bus
- [x] Safety: guardrails, redaction, side-effect levels, approval policy
- [x] Optimization: exact/semantic caching, model router
- [x] Deployment: FastAPI factory, artifact generation
- [x] CLI: init, run, evaluate, serve, inspect, plugins, doctor, version
- [x] Test suites: unit, contract, integration, security, performance
- [x] Quality gates: ruff, mypy --strict, pytest all green

## 0.3 — retrieval & ops depth

- [ ] Query rewriting + context compression in RAG
- [ ] Incremental index updates and access-control filters
- [ ] GraphRAG community summaries (hierarchical graph clustering)
- [ ] Graph database adapters (Neo4j) via the GraphStore interface
- [ ] Human-approval workflow node (interactive approver)
- [ ] OpenTelemetry SDK exporter bridge
- [ ] `aire doctor` provider connectivity checks (live mode)
- [ ] Prompt optimization loops (evaluation-guided)
- [ ] Postgres/pgvector store, Redis cache backend
- [ ] Gateway: semantic cache in front of routes, response logging sinks

## 0.4 — training & optimization

- [ ] PyTorch trainer adapter (lazy `aire[training]`)
- [ ] PEFT/LoRA fine-tuning interface, checkpoint resume
- [ ] Hyperparameter search orchestration
- [ ] Quantization/distillation adapter interfaces
- [ ] Cost-optimization policies in the model router

## 0.5 — multimodal depth

- [ ] Document understanding pipeline (PDF → structured)
- [ ] Voice agent reference pipeline (ASR → agent → TTS)
- [ ] Image generation + object detection adapters
- [ ] Video summarization pipeline

## 0.6 — scale

- [x] Multi-agent communication protocol (agent-as-tool standard) — shipped in 0.2
- [ ] Distributed workflow workers
- [ ] Scheduled/event-driven workflow execution
- [ ] Local web UI for runs, traces, costs, evaluations

## 1.0 — stability

- [ ] Public API freeze + compatibility test matrix (Python 3.11/3.12/3.13)
- [ ] Security review + dependency vulnerability scanning in CI
- [ ] Performance benchmarks published per release
- [ ] Migration guides, full guides/API reference docs
- [ ] Release candidates → 1.0
