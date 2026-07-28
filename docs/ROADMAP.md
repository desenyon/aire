# aire — Roadmap

## 0.1 (current) — architecture proof

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

## 0.2 — hardening

- [ ] Query rewriting + context compression in RAG
- [ ] Incremental index updates and access-control filters
- [ ] Human-approval workflow node (interactive approver)
- [ ] OpenTelemetry SDK exporter bridge
- [ ] `aire doctor` provider connectivity checks (live mode)
- [ ] Prompt optimization loops (evaluation-guided)
- [ ] Postgres/pgvector store, Redis cache backend

## 0.3 — training & optimization

- [ ] PyTorch trainer adapter (lazy `aire[training]`)
- [ ] PEFT/LoRA fine-tuning interface, checkpoint resume
- [ ] Hyperparameter search orchestration
- [ ] Quantization/distillation adapter interfaces
- [ ] Cost-optimization policies in the model router

## 0.4 — multimodal depth

- [ ] Document understanding pipeline (PDF → structured)
- [ ] Voice agent reference pipeline (ASR → agent → TTS)
- [ ] Image generation + object detection adapters
- [ ] Video summarization pipeline

## 0.5 — multi-agent & scale

- [ ] Multi-agent communication protocol (agent-as-tool standard)
- [ ] Distributed workflow workers
- [ ] Scheduled/event-driven workflow execution
- [ ] Local web UI for runs, traces, costs, evaluations

## 1.0 — stability

- [ ] Public API freeze + compatibility test matrix (Python 3.11/3.12/3.13)
- [ ] Security review + dependency vulnerability scanning in CI
- [ ] Performance benchmarks published per release
- [ ] Migration guides, full guides/API reference docs
- [ ] Release candidates → 1.0
