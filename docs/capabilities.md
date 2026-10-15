# Capability contracts for 0.4

This is the canonical map from the public facade to implementation and evidence.
“Contract tested” means deterministic happy and failure paths through Aire.
“Local backend tested” adds real execution against the versions below. Neither
means production certification, pretrained quality, or universal compatibility.
The facade lists 23 namespaces; the project builder is a separate entry point.

| Namespace | Supported contract | Evidence | Limits / maturity |
| --- | --- | --- | --- |
| `models` | Generation, complete tool-call streaming, embeddings, exact/semantic cache, routing | `test_stream_protocol.py`, `test_cache_isolation.py`, provider transport tests | Contract tested; paid provider availability/quality not live verified; semantic caching is approximate and text-only within exact context |
| `data` | Loading, chunking, validation, dedup, splits, label-sensitive versions | `test_chunking.py`, `test_core_quality.py`, `test_evaluation.py` | Contract tested; materializes datasets in memory; metadata must be JSON serializable |
| `rag` | Ingest, retrieve, ACL/equality filters, citations, reindex/replacement | `test_rag_offline.py`, `test_adapter_contracts.py`, reliable_local example | SQLite/local and five remote stores tested; remote nontransactional limits below |
| `graph` | SQLite/Neo4j triples, graph-grounded retrieval, lexical/model extraction, communities | `test_product_flow.py`, `test_adapter_contracts.py`, graph examples | Local backend tested; lexical co-occurrence and label propagation are heuristics |
| `memory` | Windowed short/long-term messages, restart/recall, filtering | `test_durable_composition.py`, memory example | Contract tested; local JSONL single-writer persistence; no distributed transactional memory |
| `mcp` | Stdio server/client, Streamable HTTP client, tools/resources/prompts, roots/sampling/progress | `test_mcp_transports.py`, `test_mcp_tool_contract.py` | Protocol subset; deadlines, authentication/session and callbacks tested; no full host or event-resumption service |
| `agents` | Tool loop, sessions, planning/verification, approvals, budgets, cancellation, event/token streams, teams/topologies | `test_execution_control.py`, `test_agent_offline.py`, `test_session.py`, agent_tools example | Contract tested; one direct Agent is one conversation; HTTP defaults to isolated conversations; cooperative cancellation |
| `skills` | Tool/prompt packs, executable sibling modules, static inline references | `test_plugin_contract.py`, `test_core_quality.py` | Contract tested; inline JSON tools return static text; module imports execute trusted Python |
| `workflows` | Branches, joins/cycles, bounded retries, checkpoints, approval and recovery | `test_workflow_recovery.py`, `test_reliability.py`, `test_durable_composition.py` | Contract tested; single process checkpoint ownership; bump definition_version for semantic code changes |
| `training` | Function trainer, tiny causal LM training, LoRA save/resume, real teacher/student distillation, explicit dry-run/quantization contracts | `tests/optional/test_foundation_training.py`, `test_lora_training.py`, `test_distillation.py` | Real CPU/tiny-model tests; foundation families are toy approximations, not model-compatible pretrained implementations; GPU bitsandbytes not verified |
| `ml` | Preprocessing, fit/predict/evaluate/save/load via sklearn, Torch, Keras, XGBoost, LightGBM, CatBoost; architecture blocks | `tests/optional/test_ml_backends.py`, `test_core_quality.py` | Local backend tested; skops/native formats; custom unsafe serialized types are refused; model quality is caller-owned |
| `observe` | Metrics, traces, analytics, OTLP export/flush, evaluation and fail-closed gates | `test_analytics_cli.py`, `test_telemetry.py`, `test_evaluation.py`, eval_gates example | Contract tested; zero-cost/latency values can be unknown/estimates; lexical metrics retain lexical names; exporter is best-effort, not a durable telemetry queue |
| `deploy` | API factory, indexed project serving, generated version-pinned inputs/hashed locks, Docker/Compose/K8s templates | `test_deployment_contract.py`, `test_packaging.py`, generated-container QA | Local Docker build/request/outage/recovery tested; Compose/K8s are templates, not managed hosted infrastructure |
| `gateway` | OpenAI/Anthropic request translation/SSE, aliases, fallback, auth/rate limits, policy, readiness/ownership | `test_gateway_requests.py`, `test_anthropic_gateway.py`, `test_stream_guardrails.py`, gateway_offline example | Contract tested; output rails buffer text until validated; latency/cost budgets measured after provider reporting; diagnostics require configured auth |
| `safety` | Guardrails, PII redaction, policy/approval, sandboxed filesystem tools | `tests/security`, `test_policy.py`, `test_eval_safety_035.py` | Contract tested; heuristic rails are not a complete security classifier; hard termination requires process isolation |
| `schedule` | Positive intervals, serialized ticks, job failure records, shutdown | `test_storage_recovery.py` | Contract tested; in-process scheduling, no distributed lease/clock guarantee |
| `workers` | In-process, local file queue, Redis claim/result/ack/recover | `test_storage_recovery.py`, `test_reliability.py`, live Redis interrupted-claim test | Real Redis tested; at-least-once effects, idempotency required; recover only with workers stopped; SQS unavailable explicitly |
| `recipes` | Discoverable templates for RAG/agents/eval/gateway/workflows | `test_core_quality.py`, CLI scaffold/product-flow test | Contract tested; recipes generate code/config and do not provision services |
| `locks` | Versioned application model/store references and roundtrip/apply | `test_product_flow.py`, `test_storage_recovery.py` | Contract tested; aire.lock is distinct from uv.lock and hashed Python requirements |
| `vision` | Vision requests, image-generation transport, video frame conversion; explicit detection/text stubs | `test_media_transports.py`, real FFmpeg JPEG test | Conversion locally tested; paid image/vision quality not live verified; no YOLO/full video generation |
| `audio` | ASR/TTS request/delegation, typed audio content, explicit text fallback | `test_media_transports.py`, `test_multimodal_media.py` | Transport contracts tested using valid fixtures; hosted audio synthesis/recognition not live verified |
| `docs` | PDF text extraction, scanned-PDF OCR, dataset/content conversion | `tests/optional/test_media_contracts.py`, PDF missing/empty tests | Real Tesseract/PDFium OCR tested; requires OCR extra and system Tesseract; extraction is not guaranteed for every PDF |
| `synthetic` | Structured QA augmentation with exact requested count and source provenance | `test_evaluation.py` | Contract tested; malformed/incomplete output fails; generated content quality requires evaluation |

`AI.project` is tested through index → cited answer → evaluate → passing/failing
gate → HTTP deployment in `test_product_flow.py`. `aire init` → `aire serve`
indexes its configured sources during app startup. Frozen 0.3 exports remain
covered by `test_api_freeze.py`.

## Verified local profiles

QA on macOS arm64 / Python 3.13 uses bounded CPU threads
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KERAS_BACKEND=torch`). The locked
versions include sklearn 1.9, Torch 2.14.1, Keras 3.15.1, XGBoost 3.4.1,
PEFT 0.21.2; consult `uv.lock` for exact platform-specific dependencies. CI also
runs Linux/macOS and the supported Python 3.11–3.13 core profiles, minimum core
dependencies, isolated optional profiles and installed distributions.

Real CRUD/filter/replace/delete checks ran against Qdrant 1.13.6, Chroma 1.0.20,
Weaviate 1.28.8, Milvus 2.6.2 and PostgreSQL 17/pgvector. Neo4j 5.26 and Redis
8.2 passed graph and durable queue/cache checks. Pinecone has deterministic
transport tests; no paid live validation is claimed. See [remote filtering and
replacement](rag.md#remote-filtering-and-replacement) for query-size limits and
nontransactional remote replacement.

OTLP JSON timestamp encoding follows the [OTLP specification](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding).
Export failures increment counters without masking application errors; buffering
is in memory and a crashed process can lose pending spans. Cached generation
replays have zero new usage; semantic embedding lookup has separate provider cost.

## Release boundary

A version bump is not publication. Only successful artifact, release and Pages jobs
establish those external outcomes. The repository audit preserves the original
findings; the [backlog](https://github.com/desenyon/aire/blob/main/GAPS.md) and
[release QA record](audits/2026-10-14-release-qa.md) track current evidence.
