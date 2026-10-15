# Migration guides

## 0.4.2 (from 0.4.1)

Upgrade with `python -m pip install aire-ai==0.4.2`. Runtime requirements and
published extras are unchanged.

- Explicit missing config files/directories now raise `config.not_found`.
  Omit the path to use implicit discovery/defaults. `aire inspect config` masks
  known secret-bearing keys recursively; list custom secret keys in
  `observability.mask_fields`. Arbitrary secret text cannot be inferred by key.
- Dataset inputs, returned records and lineage are isolated copies. To change
  records, return the modified record from `map` and use the returned dataset.
  Slices preserve tuple results with deep-copied records. Callback side effects
  cannot mutate the source. Sampling accepts a count or
  a fraction, not both; negative/noninteger counts and nonfinite/out-of-range
  fractions raise `data.sample_invalid`. A zero fraction now returns zero rows.
  Invalid split fractions raise `data.split_invalid` before partitioning.
- Nested/overlapping uses of one `ExecutionContext` now restore each task's prior
  ambient context. This does not make its shared usage counters independently
  budgeted. Scheduler replacement removes old handlers only after validation;
  timing intervals must be finite and positive.
- `aire doctor` keeps its JSON `checks`, `healthy`, `live` and `skipped` fields,
  and adds schema/version/Python, smoke and per-check status/required fields.
  Unconfigured credentials are explicitly skipped. Default execution is offline;
  `--smoke` runs deterministic local outcomes, `--output` saves the same JSON,
  and `--config` validates an explicit path. `--live` also probes configured
  providers and local Ollama and may incur provider usage. Failures return 1.
- Contributors can run `make audit` for the same core gate as CI. Provision
  optional dependencies/binaries before `make audit-optional`; absent prerequisites
  fail rather than certifying skipped backend tests. See [repeatable audits](audits/2026-10-15-042-audit.md).

## 0.4.1 (from 0.4.0)

This patch fixes publication automation: publisher-generated attestations use a
separate upload copy, preserving the verified distribution bundle and GitHub
assets. Runtime APIs and dependencies are unchanged. Upgrade with
`python -m pip install aire-ai==0.4.1`.

## 0.4.0 (from 0.3.5)

Existing `AI` facade calls, offline defaults, provider constructors, and the
`aire.models.types.ToolCall` import remain supported. The distribution is `aire-ai`; use `python -m pip install aire-ai==0.4.0`.
See [publishing](publishing.md) for the verified release flow.

### Conversation and streaming data

- `Message` adds `tool_calls: list[ToolCall]`, defaulting to an empty list. Existing
  JSON sessions still load. Old sessions that omitted declarations cannot recover
  those missing declarations; start a fresh conversation if a provider rejects
  orphan historical tool results.
- `GenerationResult.message` includes tool declarations. Agent memory persists
  declarations with their observations and avoids re-appending recalled history.
  A failed agent result no longer marks its session completed.
- OpenAI/Anthropic argument fragments are buffered. A `GenerationChunk.tool_calls`
  entry is now a complete call emitted once, not a partially decoded argument
  fragment. Malformed JSON and incomplete tool streams raise `ProviderError` with
  `provider.stream_tool_invalid` or `provider.stream_incomplete`.
- Exact cache replay retains tool calls. `SemanticCachedModel` bypasses requests
  containing tools or tool history; approximate prompt matches cannot identify
  equivalent tool arguments.

### Index publication

`Knowledge.reindex_document` prepares and validates new chunks before changing
stored data. Local replacement swaps the in-memory mapping; SQLite replacement
commits one database transaction before updating its cache. Invalid embedding
counts, dimensions, or non-finite values produce `rag.invalid_embeddings`.
OpenAI-compatible embedders with unknown model names report dimension zero until
their first valid response, then retain the discovered dimension. Known OpenAI
embedding models keep their declared dimensions.

`Knowledge.ingest(..., replace=True)` is additive. `IncrementalIndex.reindex` with
`clear=True` uses it instead of clearing first. Local/SQLite implement atomic full
replacement. A third-party adapter without `replace_all` now receives
`rag.atomic_replace_unsupported` before any clearing. Add an atomic override or
manage a backend-specific staging collection and swap. `clear=False` still ingests
without clearing. Empty ingestion remains an error; explicitly call `clear()`
when deletion is your intent.

The generic `replace_document` fallback enumerates up to one million empty-query
hits, writes new chunks, then deletes obsolete IDs. This preserves old data on an
embedding/upsert failure but does not guarantee transactional service updates or
complete remote enumeration. Prefer adapter-specific replacement where available.

### Persistence and workers

`write_json_file` and `write_jsonl` now write a private temporary file in the target
directory, flush/fsync it, and use `os.replace`. They leave the old snapshot intact
on serialization/write/replace failure and remove temporary files. Existing JSON
formats are unchanged apart from the new message/workflow fields. Replacement
files use private temporary-file permissions; account for that if another user
previously read a group-readable snapshot. Concurrent writers remain
last-writer-wins; directory/power-loss durability is not guaranteed.

File queues acknowledge only after saving `done/<id>.json` or `failed/<id>.json`.
Invalid payloads, cancellation, and I/O failures retain `.json.claimed` files.
With **all workers stopped**, inspect/repair malformed jobs and call `recover()`.
Claims with an existing result are removed without replay. Others are requeued.
Repeated side effects remain possible after a crash before acknowledgement.

Redis queues add `<key>:processing` (list) and `<key>:results` (hash). Drain/stop old
workers before switching implementations. The existing queued `Job` JSON and
original key remain compatible. Claims use `BRPOPLPUSH`; acknowledgements use Lua
to write the result before removing a claim. Stop workers and wait for outstanding
blocking calls before `recover()`. Recovery requeues processing items; it does not
check leases. Retain/delete result history according to your own policy. Redis
Cluster hash-slot management is not implemented.

### Workflow checkpoints

New checkpoints carry `scheduler_version=1`, nested edge firing/consumption counts,
and pending/in-flight nodes. Node visit budgets survive resume. Untaken branch
chains reach a fixed point before joins are scheduled, exhausted failures cannot
turn into successful empty runs, and disconnected pending nodes report a stall.
Closing a stream cancels its unfinished tasks.

Older checkpoints without scheduler fields use the legacy reconstruction path.
DAG resume remains supported. Exact cyclic resume requires a new-format checkpoint
because old records did not persist historical edge decisions. Resume with the
same graph/functions, use one checkpoint path per run, and make side effects
idempotent: tasks that finished before their record was saved can run again.

### Development environment

The default dev environment does not install PyTorch. Its foundation test checks
that the optional dependency is requested correctly when absent, and constructs
the tiny model when present. Strict typing tolerates absent optional Torch/OTel
imports without ignoring other project typing errors. The Ruff formatting baseline
and case-sensitive MkDocs paths are repaired. CI adds the offline integration smoke
and, on Python 3.12, strict docs plus wheel/sdist builds.

## 0.3.4 → 0.3.5

Honesty + depth release. Most public names remain; behaviour becomes stricter.

| Change | Action |
|--------|--------|
| Guardrails auto-wire on `Knowledge.ask` / `create_gateway` | Pass `guardrails=False` to opt out; tune via `SafetyConfig` |
| `bleu` metric | Now BLEU-4 + brevity penalty; old F1 approx is `bleu_approx` |
| `cross_encoder` reranker | Defaults to HF CrossEncoder (`aire-ai[eval]`); pass `model=` LLM for old scorer, or use `reranker="model"` |
| `embedding_similarity` / `nli_faithfulness` / `model_judge` | Raise `ConfigurationError` when judge/embedder missing (no silent 0.0) |
| Gateway images | Returns **501** without image capability |
| Workers | `create_worker("redis")` available; `"sqs"` raises until boto3 is bundled |
| Foundation | `foundation` remains toy; use `foundation_pretrained` / `from_pretrained` for HF weights |
| Scale pack Postgres | `AIRE_DATABASE_URL` auto-wires `PgVectorStore` Knowledge in generated `app.py` |
| Anthropic `/v1/messages` | Response emits real `tool_use` blocks; request maps `tool_result` → `role=tool` |

## Pre-0.3.4

See [CHANGELOG](https://github.com/desenyon/aire/blob/main/CHANGELOG.md). Prefer upgrading to the latest 0.3.x patch before jumping minors.

## 0.4 remote index identity

Qdrant and Weaviate now derive stable UUIDs from original Aire chunk IDs, and
remote adapters retain the document ID and chunk position. Reindex existing
collections into a fresh collection before switching traffic. Earlier Qdrant
numeric IDs used Python's process-randomized hash and cannot be recomputed safely
after restart; preserve original `aire_id` payloads if migrating them manually.

Remote mutations now raise normalized provider errors for HTTP failures and
Weaviate/Milvus per-operation errors. Document replacement refuses to write when
the backend cannot enumerate the complete existing index. Qdrant, Chroma and
Milvus text scans page through records; Pinecone enumeration requires its
serverless list/fetch APIs. Replacement remains best effort across remote writes
and deletes; use collection/version swaps for transactional publication.

## 0.4.0 deployment and state contracts

- The version comes from `src/aire/_version.py`; `pyproject.toml` reads it dynamically.
- Generated loose dependency files are now `requirements.in`, with an exact Aire
  version. `aire.deployment.lock_requirements(directory, python_version="3.12",
  python_platform="linux")` resolves `requirements.lock` with hashes using uv and
  switches the generated Dockerfile only after success. `aire.lock` still stores
  application references; it is not a Python dependency lock.
- HTTP agent/team runs use fresh conversation memory for each request. Explicit
  `create_app(..., shared_agent_session=True)` preserves a single shared conversation
  and serializes requests. Put authenticated tenant/session routing in your application
  when you need multiple durable conversations.
- Generated apps fail startup when a configured backend cannot initialize. An explicit
  `AIRE_ALLOW_OFFLINE_FALLBACK=1` enables the offline fallback for development.
- Workflow checkpoints reject changed node/edge definitions. Set and bump
  `Workflow(..., definition_version="...")` when function semantics change without a
  graph-shape change. File checkpoints and JSONL memory need one process owner;
  same-process overlapping workflow checkpoint owners are rejected.
- Cache replays charge zero new generation tokens/cost; `raw.cache_hit` and
  `raw.source_usage` identify original usage on full results. Semantic lookup embedding
  costs are separate and are not included in generation billing.
- sklearn state now uses skops; boosting adapters use native JSON/UBJ/CBM formats.
  No arbitrary pickle loading or automatic trust of serialized custom classes. Refit
  old states that did not persist an underlying model; cross-version backend compatibility
  follows the backend format's limits.
- Dataset versions now include labels/metadata; text deduplication remains text-based.
  Synthetic generation raises if the model returns fewer QA pairs than requested.
