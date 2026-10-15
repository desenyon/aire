# Aire capability completion implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fulfill all documented capabilities with reliable packaging and honest limits.

**Architecture:** Repair public contracts in place, then consolidate repeated internals.
Preserve the facade and lazy optional adapters. Evidence, supported scope and release gates
live in one capability matrix rather than independent completion claims.

**Tech Stack:** Python 3.11–3.13, Hatchling, Pydantic, httpx, Typer, pytest, Ruff, mypy,
MkDocs, GitHub Actions; optional ML/storage/media backends.

**Spec:** [Capability completion design](../specs/2026-10-14-capability-completion-design.md).
**Evidence:** [Repository audit and GitHub PR matrix](../../audits/2026-10-14-repository-audit.md).

## Global constraints

- Distribution: `aire-ai`; import: `aire`; CLI: `aire`.
- Python: preserve tested 3.11, 3.12, 3.13 support; establish macOS arm64 and Linux verification.
- Preserve frozen 0.3.x exports. Breaking removals require a minor release and migration notes.
- Core stays offline-capable with four runtime dependencies and lazy optional integrations.
- Existing public extras and provider-named no-op extras remain compatible during cleanup.
- Credentials never appear in fixtures, traces, locks, issue bodies, or examples.
- Real backend success requires real backend evidence; dry runs do not establish model quality.

## Review focus

1. Restricted output split across stream chunks: no prohibited bytes may escape (Task 3).
2. Same chunk ID after process restart: upsert and deletion refer to the same remote object (Task 5).
3. Cancellation/timeout during a side effect: no retry duplicates the side effect (Task 4).
4. Base wheel installed outside the checkout: no optional import or source-tree dependency (Task 2).
5. Dependency outage at startup: configured deployment cannot silently become a mock service (Task 6).

## Iterative loop and stopping rule

For each task: reproduce → add failing test → implement one contract repair → run the targeted
suite → run affected integration/example → update capability evidence → review diff → commit/PR.
Repeat against a fresh audit of that boundary, including adversarial and malformed inputs.
Keep each PR independently releasable. Refactors follow regression tests, not precede them.

A task is complete only when its listed tests pass and the public docs/describe output agree.
A milestone closes only after all its P0/P1 findings are closed; new findings enter the same
tracker with a reproducer and severity. A passing suite or a green constructor test alone
cannot close a capability. Unverified optional backends remain visibly unverified.

## Sequence

Task 1 is implemented in this branch. Tasks 2–6 are the next release blockers. Tasks 7–11
complete the advertised surface. Task 12 applies the final release gate. Task 2 can precede
runtime repairs without changing their public contracts. No dependency PR is auto-merged.

### Task 1 — Correct distribution identity (F01; implemented)

**Files:** `src/aire/deployment/{artifacts,scale}.py`, install hints across source/docs/examples,
`docs/publishing.md`, `tests/unit/test_packaging.py`, existing missing-extra tests.
**Interfaces:** Preserve `generate_artifacts(...) -> DeployArtifacts` and
`generate_scale_pack(...) -> ScaleArtifacts`; generated requirements target `aire-ai`.

- [x] Add tests parsing generated requirements and comparing distribution metadata, preserving custom requirements and sidecar extras.
- [x] Observe both new tests fail against `aire[...]` output.
- [x] Correct install names; keep `import aire` and the `aire` executable unchanged.
- [x] Verify 125 tests pass, 5 skip; lint, formatting and strict types pass.
- [ ] Release the repaired package through Task 12. Current source changes are not yet on PyPI.

### Task 2 — Make installation and dependency maintenance reproducible (F02, F17)

**Files:** `pyproject.toml`, new `uv.lock`, `src/aire/_version.py`, deployment generators,
`.github/workflows/ci.yml`, `docs/{publishing,compatibility,contributing}.md`;
new `tests/contract/test_distribution.py` and packaging smoke script.
**Interfaces:** Preserve published extras. `aire.lock` remains an application-reference file;
Python dependency locks have a separate explicit format and platform target.

- [ ] Add wheel-install subprocess tests: metadata version equals `aire.__version__`, CLI entrypoint works, `py.typed` and provider entrypoints are present, optional stacks remain unloaded.
- [ ] Add deployment-install test using a freshly built wheel in a temporary environment, outside the repo; preserve existing user `app.py` and custom requirements.
- [ ] Introduce test/docs/release dependency groups and commit a tested lock; keep published compatibility extras during transition. Resolve all declared profiles and explicitly encode supported platform conflicts.
- [ ] Replace duplicated aggregate constraints only after testing extra-equivalence in built metadata; centralize version provenance. Test oldest supported versions separately from the lock.
- [ ] Generate exact deployment dependencies for a selected platform or rename the current loose file to requirements input with migration support; never label lower bounds a lock.
- [ ] Verify `uv sync --locked` for development, base-wheel smoke on Linux/macOS, and Python 3.11–3.13 minimum/current profiles. Do not install every heavyweight extra by default.

Design references: uv separates published extras from local [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/);
[`--locked`](https://docs.astral.sh/uv/concepts/projects/sync/) rejects stale project locks.

### Task 3 — Unify gateway output policy and cache isolation (F03, F09, F10)

**Files:** `src/aire/deployment/gateway.py`, `src/aire/optimization/{cache,redis_cache}.py`,
`src/aire/safety/guardrails.py`; new `tests/security/test_stream_guardrails.py`,
`tests/unit/test_cache_isolation.py`; `docs/gateway.md`.
**Interfaces:** Preserve `create_gateway` and model `generate/stream`; add optional cache
namespace only if caller identity is part of the supported cache contract.

- [ ] Add failing tests for OpenAI and Anthropic streaming output block/redact, including a secret split across chunks, a provider error, and disconnect cleanup.
- [ ] Choose safe buffered output validation when output rails are enabled; document the latency cost. Preserve unguarded native streaming when explicitly configured.
- [ ] Test nested ToolCall argument mutation on stream replay cannot affect later results; deep-copy at the cache boundary.
- [ ] Test role swaps, changed system messages, different images with identical captions, different namespaces, tool contexts, and differing response formats never share unsafe semantic results.
- [ ] Centralize cache eligibility and exact context signatures for wrapper/gateway/Redis paths; only compare semantically eligible text within compatible context.
- [ ] Verify security tests, existing provider/gateway protocol suites, and cache roundtrip tests using an injected transport and real local cache objects.

### Task 4 — Share agent execution control (F04, F05, F06, F15)

**Files:** `src/aire/agents/{runtime,streaming,agent,types}.py`, `src/aire/core/context.py`,
`src/aire/tools/tool.py`; new `tests/unit/test_execution_control.py`; `docs/agents.md`.
**Interfaces:** Preserve `AgentExecutor.run`, `Agent.run`, `run_stream`, and `stream_tokens`;
one internal control path handles budgets, cancellation, state transitions and usage.

- [ ] Test zero token budget prevents provider invocation in all three paths; pre-cancelled context produces no model or tool call.
- [ ] Test cancellation after one step, provider failure, exhausted step budget, output-only final usage, and client stream closure have consistent terminal behavior.
- [ ] Test a synchronous tool does not block an async heartbeat and timeout returns before the tool completes; a timed-out write is never retried automatically.
- [ ] Move synchronous tool bodies off the event loop with documented continuation semantics; enforce cancellation at model/tool boundaries and account usage on every path.
- [ ] Remove duplicated loops only after parity tests; retain documented turn-based session persistence and separate-resume budget limits unless a new cumulative contract is implemented.
- [ ] Verify agent/config/session/policy/tool tests plus reliable-local smoke; add a deterministic side-effect counter assertion for parallel tools and cancellation.

### Task 5 — Repair remote adapter identity and failure contracts (F07, F08)

**Files:** `src/aire/integrations/{qdrant,chroma,milvus,pinecone,weaviate,http}.py`,
`src/aire/rag/{store,pgvector}.py`; new `tests/contract/test_vector_adapters.py`,
expanded `tests/live/test_live_stores.py`; `docs/{rag,migration}.md`.
**Interfaces:** Preserve `VectorStore` CRUD methods. Define stable serialized Chunk identity,
backend capability descriptions, and explicit unsupported replacement behavior.

- [ ] Test Qdrant IDs in two subprocesses with different `PYTHONHASHSEED`; upsert/delete must use identical deterministic IDs.
- [ ] Test `document_id`, chunk index, text, metadata and ID roundtrip through each adapter transport; replacing a document removes only its old generation.
- [ ] Inject HTTP 401/404/429/500, malformed data and partial batches; assert normalized failures and no success count on failed writes/deletes.
- [ ] Implement deterministic Qdrant IDs, full identity persistence, checked HTTP mutations, and paginated document enumeration or fail explicitly when unsupported.
- [ ] Document migration from hash-derived IDs; use stored original IDs/reindexing rather than assuming old Python hashes are recoverable.
- [ ] Run container-backed Aire CRUD/filter/replace/delete/reopen tests for supported services, including ACL filters before limits. Preserve backend-specific transaction limits.

### Task 6 — Make deployment startup, readiness and resource ownership truthful (F16)

**Files:** `src/aire/deployment/{artifacts,scale,fastapi_app,gateway}.py`, `src/aire/core/{runtime,lifecycle}.py`;
new `tests/integration/test_deployment_contract.py`; `docs/{gateway,ARCHITECTURE}.md`.
**Interfaces:** Preserve application factories; add explicit offline fallback configuration
with a safe default, and ownership-aware app shutdown.

- [ ] Test configured Postgres/Redis initialization failure cannot return an echo service or HTTP-ready state.
- [ ] Test degraded backend health gives a readiness failure while liveness can remain successful; validate auth on intended diagnostic endpoints.
- [ ] Validate request bounds/types return structured 4xx responses, not raw ValueError/500; test agent concurrency/session isolation.
- [ ] Implement fail-closed configuration, meaningful readiness, and lifespan shutdown for owned clients; never close externally owned resources unexpectedly.
- [ ] Build generated container from the tested wheel, send a real request, exercise dependency outage/recovery, and verify graceful shutdown.

### Task 7 — Verify durable composition and memory/scheduling boundaries

**Files:** `src/aire/{workflows,workers,memory,schedule}/`, `src/aire/agents/{session,memory}.py`;
existing recovery tests and new `tests/integration/test_durable_composition.py`.
**Interfaces:** Preserve Workflow/Worker/Session public results and at-least-once semantics.

- [ ] Extend existing failure-injection coverage across claim, execution, result persistence, acknowledgement, recovery and repeated recovery; assert stable job IDs and no lost work.
- [ ] Test concurrent workflow instances, cyclic/conditional joins, graph-definition changes on resume, cancellation and per-run checkpoint ownership.
- [ ] Test long-term memory restart/recall with tool message integrity and tenant filters; test scheduler overlap, negative/zero intervals, failure records and shutdown.
- [ ] Repair demonstrated gaps; describe local-only file queues and multi-process SQLite cache limitations. Do not promise exactly-once side effects.
- [ ] Run local recovery suite and real Redis queue crash/recovery integration before promoting Redis support.

### Task 8 — Make evaluation and observability results dependable (F11, F14)

**Files:** `src/aire/{evaluation,observability,optimization}/`, `src/aire/cli/main.py`;
new `tests/unit/test_analytics_cli.py`, `tests/contract/test_evaluation.py`, OTLP fixture tests.
**Interfaces:** Preserve metric names/aliases; separate measured, estimated, synthetic and
unsupported values in output. Resolve `AIRE_ANALYTICS` as a documented module attribute.

- [ ] Test nonexistent analytics reference fails clearly; synthetic data always has `demo_mode=true`; real source survives JSON/Prometheus output.
- [ ] Test empty datasets, zero denominators, missing metric gates, failed evaluation cases, judge/embedder absence and cached billing/token accounting.
- [ ] Test OTLP payload hierarchy and flush-on-close with transport failures; telemetry failure cannot mask application errors.
- [ ] Fix measured defects and extract shared metrics/error helpers; retain explicit lexical metric names and documented aliases.
- [ ] Verify evaluated-system → gate → deployment flow on deterministic data; keep model quality claims separate from plumbing tests.

### Task 9 — Complete optional ML, training, media and document contracts

**Files:** `src/aire/{ml,training,audio,vision,docs,multimodal}/`, `pyproject.toml`;
new optional-backend contract suites and real small input fixtures.
**Interfaces:** Keep adapter exports and documented stubs; each result indicates actual
backend execution versus dry-run or placeholder output.

- [ ] Build a minimal fixture for each advertised backend: sklearn/boosting fit-predict-save-load, Torch/Keras training, LoRA save-resume, distillation, quantization, image/audio/video/PDF/OCR conversion.
- [ ] Test missing extras/binaries produce actionable errors with `aire-ai` names; exercise empty/corrupt inputs and unsupported model capabilities.
- [ ] Test original methods through Aire, not just upstream imports; a tiny locally constructed model can verify training without downloading pretrained weights.
- [ ] Repair demonstrated gaps, version floors, backend selection and serialization. Validate real media bytes when `stub=False`; refuse fabricated-success outputs.
- [ ] Record platform/backend versions and live/manual evidence separately. Keep full foundation pretraining, hosted serving, YOLO and full video generation out of scope unless separately specified.

### Task 10 — Verify MCP, plugins, tools and generated schemas

**Files:** `src/aire/{mcp,tools}/`, `src/aire/agents/{skills,toolkits}.py`,
`src/aire/core/{plugins,registry}.py`; new subprocess/HTTP contract tests; `docs/mcp.md`.
**Interfaces:** Preserve stdio/HTTP subset and entrypoint groups; capability negotiation
must match implemented handlers and errors.

- [ ] Test actual stdio subprocess and HTTP transport initialize/list/call/resource/prompt roundtrips, deadlines, malformed frames, cancellation, auth and closure.
- [ ] Test supported roots/sampling/progress callbacks and unsupported-method errors; publish exact protocol version/subset in docs and describe output.
- [ ] Test OpenAPI nested schemas, required inputs, safe URL handling, tool permission/approval boundaries and plugin collisions/import failures.
- [ ] Remove placeholder skill execution claims or implement the stated behavior; keep descriptive skill content explicitly descriptive.
- [ ] Verify wheel-discovered providers and third-party plugin fixtures without globally mutating runtime state.

### Task 11 — Consolidate capability documentation and CLI examples (F13, F14, F15)

**Files:** new `docs/capabilities.md`, `README.md`, `GAPS.md`, `docs/{honesty,product,PUBLIC_API,cli,GUIDE,compatibility}.md`,
`src/aire/ai.py`, `src/aire/cli/main.py`, examples, `mkdocs.yml`; new CLI tests.
**Interfaces:** Preserve all 23 `AI.describe()['namespaces']` entries and frozen exports.
A single capability matrix records operations, maturity, limits, evidence and ownership.

- [ ] Inventory every documented operation and link its test/example; classify supported, experimental, intentionally unsupported or unverified. Bugs stay open.
- [ ] Test CLI init/doctor/deploy/evaluate/analytics/manifest through Typer's runner and wheel subprocesses; test read-only and failing exit paths.
- [ ] Convert examples to assert their named outcome: calculator executes, gateway handles a request, evaluation gate blocks failure, durable workflow resumes.
- [ ] Reconcile MCP/Pages/stub claims, remove duplicate status lists, link documentation to the canonical capability matrix; preserve migration history.
- [ ] Extract facade namespaces/shared CLI output only where tests protect identity and signatures. Remove no-op aliases only through a documented deprecation process.
- [ ] Run all offline examples in temporary directories plus strict docs and link verification.

### Task 12 — Refresh GitHub PRs and gate the next release (F12, F14, F17)

**Files:** `.github/workflows/{ci,docs,release,benchmarks,codeql}.yml`, `.github/dependabot.yml`,
`docs/publishing.md`, `CHANGELOG.md`; repository Pages settings.
**Interfaces:** Every release artifact is tied to a tested SHA and matching version/tag.

- [ ] Refresh PRs #1–#12 against repaired main and execute each acceptance check in the audit's PR matrix. Record results individually; do not call old failures current incompatibility evidence.
- [ ] Test action majors and upload/download compatibility in workflow runs; group future updates by ecosystem without hiding backend compatibility checks.
- [ ] Verify Pages settings/access, then deploy and inspect the public URL; record exact deployed commit.
- [ ] Require base-wheel tests, relevant optional-backend tests, locked environment, strict docs, supported Python/OS profiles and triaged dependency audit before release.
- [ ] Add meaningful benchmark regression thresholds after establishing repeatable baselines; avoid timing gates from one noisy run.
- [ ] Check tag/project/runtime versions agree; publish only artifacts built from validated code. Use a dry run before the next version, then record fresh install proof.

## Capability coverage map

| Public namespaces / contracts | Owning tasks |
| --- | --- |
| models, routing/retry, integrations, base import | 2, 3, 12 |
| data, rag, index/filter/citation contracts | 5, 7, 11 |
| graph, community/extraction/GraphRAG | 5, 7, 11; include graph failure/roundtrip fixtures |
| agents, tools, approvals, teams/topologies, skills | 4, 7, 10 |
| memory, workflows, workers, schedule, locks | 2, 7 |
| safety, gateway, deploy | 3, 4, 6 |
| evaluation via facade, observe, optimization, synthetic | 3, 8, 11; include generated-dataset provenance/errors |
| ml, training | 2, 9 |
| vision, audio, docs, multimodal | 9 |
| mcp, plugins | 10 |
| recipes, project builder, CLI, all facade exports | 6, 11 |

Additional Task 7 fixtures cover `KnowledgeGraph.ingest/ask`, triple persistence/community
queries and application-reference lock roundtrips (including unknown schema versions and
store-pin limitations). Task 8 covers synthetic dataset size/schema/provenance failures;
Task 11 covers each recipe and the project-builder index/ask/evaluate/deploy path.

## Release completion checklist

- [ ] All P0/P1 findings resolved with regression evidence.
- [ ] Every supported operation in the capability inventory has behavioral evidence and accurate docs.
- [ ] Optional backends carry tested versions/platforms and explicit remaining limitations.
- [ ] Installation, dependency update, built-wheel and generated-deployment contracts pass.
- [ ] All GitHub review items have verified disposition; Pages and release dry run succeed.
- [ ] No synthetic/stub/demo result is presented as a real execution or measured quality result.
