# Repository audit — 2026-10-14

This is the historical baseline audit. For repaired 0.4.0 behavior and current
validation, see the [release QA record](2026-10-14-release-qa.md) and
[capability matrix](../capabilities.md). Findings below retain their original
observations and acceptance criteria.

Baseline: `70b4633e94183961347b7acef5be06b938bb22dc` on `main`.
Scope agreed with the maintainer: **Fulfill all documented capabilities, with reliable packaging and honest limits.**
This is a source, contract, packaging, and GitHub audit, not a certification that every defect has been found.

## Verified baseline

| Check | Result |
| --- | --- |
| Working tree before audit | Clean |
| macOS arm64, Python 3.13.16 | Local test platform |
| Offline pytest | 123 passed, 1 skipped, 4 live tests deselected |
| Ruff lint / formatting | Passed; 221 Python files formatted |
| Strict mypy | Passed; 187 source files |
| Coverage with branches | 34.03% combined; 37.76% statements; 18.68% branches |
| Reliable-local integration example | Passed |
| All example `main.py` programs | 14 exited 0 in a temporary working directory, live opt-ins removed |
| `aire doctor` | Healthy offline; credentials absent, as expected |
| Strict MkDocs build | Passed locally |
| Isolated wheel + sdist build | Passed with Hatchling 1.32.4 |
| Base wheel in fresh environment | Import, echo model, plugin discovery, doctor, dependency check passed |
| Heavy imports after base smoke | None of torch, tensorflow, keras, sklearn, pandas, fastapi, numpy loaded |
| GitHub current-main CI | Success, run 37872300438 |
| GitHub Docs | Build succeeded, deploy failed 404, run 37872300409 |

The complete suite after the package-name repair reports **125 passed, 5 skipped**.
No optional ML stack, remote service, paid provider, GPU, or Docker deployment was validated live.
The four live tests currently probe connectivity, not Aire adapter CRUD or recovery.
Coverage is evidence of unexercised paths, not a correctness percentage.

## Reproduce targeted findings

Run `python docs/audits/reproduce_2026_10_14.py` in the development environment.
The [probe script](reproduce_2026_10_14.py) uses local models and injected transports,
prints observations rather than asserting success, and contacts no external provider.
[Recorded output](2026-10-14-probe-results.json) was captured after the package-name repair;
other runtime defects remain open. The hash-seed check and analytics CLI probe are
described separately in F07 and F11.

## Findings and repair acceptance criteria

### F01 — Wrong distribution in deployment and install guidance (P1, repaired in this branch)

`deployment/artifacts.py` generated `aire[serve]>=0.1.0`; `deployment/scale.py`
added the same wrong distribution for Redis/Postgres. The distribution is `aire-ai`.
Missing-extra errors and examples repeated the wrong install target.

Changed generated requirements and all matching source/docs/example install hints to
`aire-ai[...]`. Two new tests parse real generated requirements and compare the target
to installed distribution metadata. Both failed before the fix and pass afterward.
The existing publishing verification also used nonexistent `AI.describe()['kind']`;
it now reads `['library']`.

**Still open:** the misleading `requirements.lock` filename and unbounded requirement
ranges are not reproducible deployment locks. This patch does not release a new wheel to PyPI.

### F02 — No reproducible package/development contract (P1, confirmed inspection)

`pyproject.toml` has no committed dependency lock; `all` duplicates almost every optional
requirement; Python metadata is open-ended while CI tests 3.11–3.13 on Linux only.
CI installs `.[dev]` and only installs docs on 3.12. Optional backend PRs therefore do
not prove the backend installs or executes. Runtime version and project version are
maintained separately. Generated deployment requirements accept `>=0.1.0`.

**Accept when:** supported platform/Python profiles resolve, minimum and locked supported
versions execute, built wheels are tested outside the checkout, deployment locks are exact,
and metadata/CLI/runtime versions agree. Preserve base-install lazy imports and public extras.

### F03 — Streaming bypasses output guardrails (P0, reproduced)

`deployment/gateway.py` applies `_guard_generation_result` only to nonstream responses.
A test model returning `BLOCKME`, with a guardrail rejecting that output, produced:
nonstream HTTP 400; stream HTTP 200 containing `BLOCKME`.
The same structural omission exists in the Anthropic streaming route.

**Accept when:** both protocol routes enforce the configured output policy before releasing
restricted bytes, including text split across chunks. Define buffered guarded streaming or
explicitly reject unsupported policy/stream combinations. Input checks alone are insufficient.

### F04 — Agent token streaming bypasses budgets (P1, reproduced)

`agents/streaming.py:stream_tokens` directly streams the model when there are no tools.
With `AgentConfig(token_budget=0)`, ordinary executor run returns `budget_exceeded`;
`stream_tokens` still emits `hello`.

**Accept when:** normal, event, and token-stream paths share usage/control enforcement;
zero budget makes no provider call, measured usage stops subsequent work, and closing a
stream cancels pending work. Document that post-response provider usage cannot be a hard
prebilling cap without reservation/estimation support.

### F05 — Cancelled context still executes (P1, reproduced)

`AgentExecutor.run(..., context=ctx)` completes after `ctx.cancel()`.
`ExecutionContext.check_cancelled` exists but the executor never calls it.

**Accept when:** cancellation before execution prevents model/tool invocation, cancellation
between steps prevents further side effects, and cancellation has a documented terminal
result/error across all public execution paths.

### F06 — Synchronous tool timeout does not work (P1, reproduced)

`Tool._invoke` calls a synchronous function on the event-loop thread. A function sleeping
40 ms returned success with `timeout_seconds=0.001` (observed elapsed approximately 45 ms).
`asyncio.wait_for` cannot interrupt the blocked loop.

**Accept when:** ordinary sync tools do not block the loop; timeout returns promptly;
thread continuation after timeout is disclosed and is never automatically retried. Hard
termination requires a separate process contract, not an unsafe thread-kill promise.

### F07 — Qdrant identifiers are unstable and deletes use another format (P1, reproduced)

`integrations/qdrant.py` writes `abs(hash(chunk.id)) % 2**63` but deletes raw string IDs.
For the same `stable` ID, separate Python hash seeds produced
8581078515176704784 and 2184375646213989810.
An injected transport captured a numeric upsert ID and `{"points": ["stable"]}` on delete.

**Accept when:** one deterministic encoding is used by all CRUD paths and subprocesses;
existing payload `aire_id` can be used for a documented migration/reindex procedure.

### F08 — Remote stores lose document identity / ignore failures (P1, reproduced Qdrant; wider review open)

Qdrant writes no `Chunk.document_id` or index and reconstructs a chunk with document ID `''`.
The generic replacement fallback matches `chunk.document_id`, so it cannot identify old
chunks. A fake HTTP 500 delete response returned success count 1. Chroma also has raw
unchecked write/delete calls and does not reconstruct document identity in its shown path.
Other remote adapters need the same transport-contract suite; do not assume all share
identical bugs without executing it.

**Accept when:** round-trip identity is preserved, all HTTP failure responses raise normalized
errors, listing is paginated or replacement is explicitly unsupported, and failed preparation
never removes the existing index. Backend-specific nontransactional limits remain explicit.

### F09 — Stream cache hits expose mutable cached tool arguments (P1, reproduced)

`CachedModel.stream` passes cached ToolCall objects to the replay chunk. ToolCall is frozen,
but its argument dictionary is mutable. Changing streamed `arguments['a']` to 999999 changes
later `generate()` cache hits. Usage itself is frozen and did not reproduce this issue.

**Accept when:** returned nested data cannot mutate cache entries across generation or streaming;
include tool arguments and nested metadata in mutation-isolation tests.

### F10 — Semantic cache ignores message roles and media (P1, reproduced roles; media inspection)

`SemanticCachedModel` and gateway semantic caching flatten message text and exclude messages
from the parameter signature. Swapping system/user roles with identical ordered text yields
a cache hit (reproduced). Image/audio content is also absent from the flattened text.

**Accept when:** semantic similarity applies only within an exact compatible role/system/context
boundary; media/tool/unknown context bypasses caching unless safely fingerprinted. Test both
model wrapper and gateway/Redis routes. Include caller-provided cache namespace where needed.

### F11 — Analytics can label fabricated counters as live (P2, reproduced)

`AIRE_ANALYTICS=audit_nonexistent aire analytics` returns a synthetic mock:echo request with
`demo_mode: false`. The environment value changes the label but is never imported/resolved.

**Accept when:** the reference loads and validates a real configured source or fails explicitly;
no fallback counter is labeled live. Test JSON and Prometheus CLI output.

### F12 — Pages deployment incomplete (P1, external state verified)

[Docs run 37872300409](https://github.com/desenyon/aire/actions/runs/37872300409)
built the site, then failed to create a deployment with 404. `gh api repos/desenyon/aire/pages`
also returned 404. That is consistent with missing/inaccessible Pages configuration; the API
response alone does not distinguish disabled Pages from insufficient access.

**Accept when:** Pages configuration and permissions are verified, the deployment succeeds, and
the public documentation URL serves the intended commit. Do not mark this fixed by local MkDocs.

### F13 — Completion records and capability docs disagree (P2, confirmed inspection)

The prior GAPS.md says all depth items are closed and Pages is done. `docs/mcp.md` says stdio
only and no assumed roots/sampling; implementation and GAPS list HTTP and client handlers.
`docs/honesty.md` and the README describe stubs, while the old backlog claims remaining
subsets are not unfinished stubs. `AI.describe` inventory alone is not a conformance test.

**Accept when:** each claim names implementation, scope, regression/integration evidence and
limitations. Keep one capability matrix; link overview prose to it rather than duplicating statuses.

### F14 — Validation misses public behavior (P1, confirmed measurement)

CLI, observability, project locks, six remote adapters, and several ML modules have zero
coverage in this offline run. `tests/live/test_live_stores.py` pings services or executes
`SELECT 1`/`RETURN 1`; it does not exercise Aire classes. The agent-tools example completes
without invoking its calculator. Gateway example only constructs routes. Performance CI
uploads results without a regression gate. Dependency audit is `continue-on-error`.

**Accept when:** public contracts have behavioral tests, adapter tests run actual Aire APIs,
examples assert their advertised outcome, regressions fail CI, and dependency exceptions are
explicit and time-limited. Use coverage to locate missing contracts, not to claim 100% reliability.

### F15 — Repeated control logic and oversized entry points (P2, design concern)

`ai.py` is 1,682 lines; `gateway.py` 1,471. Agent normal/event/token paths duplicate control
logic; cache wrappers duplicate replay/capture logic; HTTP API shapes duplicate validation.
These are maintenance risks with concrete divergence in F03/F04/F09, not evidence that all
large files should be rewritten.

**Accept when:** extract shared policy/control/cache contracts after regression tests, preserve
frozen imports and signatures, and remove only code proven unused or replaced.

### F16 — Configured deployment can silently become an echo app (P1, confirmed inspection)

The generated `_APP_SCAFFOLD` catches any Postgres/Redis setup exception, logs a warning,
and returns an offline echo target. The service can appear healthy while the configured
persistent workload is absent. `/ready` checks runtime presence, not backend health.

**Accept when:** explicitly configured dependencies fail startup/readiness on failure; offline
fallback requires explicit opt-in. Preserve user-owned `app.py` on regeneration.

### F17 — Release pipeline is detached from semantic validation (P1, confirmed inspection)

Release build runs build/twine, then can publish on a matching tag or manual invocation.
It does not itself require package tests, verify tag/version equality, or install its built
wheel. The existing release was v0.3.5; current reliability work is unreleased.

**Accept when:** release consumes artifacts from tested code, checks all versions/tag,
exercises installation/entry points, and cannot publish an unverified manual build.

## GitHub review disposition

There are no review threads on PRs #1–#13, and no preexisting issues at audit time.
PR #13 is merged. PRs #1–#12 remain open; all show failed July CI runs, not fresh results
on the October main branch. The sampled #12 run fails formatting (44 files), not a PEFT
compatibility test. Do not infer that all dependency upgrades are broken or safe.

| PR | Change | Required evidence before disposition |
| --- | --- | --- |
| #1 | deploy-pages v4 → v5 | Release-note/runner review plus Pages configuration and successful deployment |
| #2 | upload-artifact v4 → v7 | Release and benchmark upload/download roundtrip; check download-artifact pairing |
| #3 | codeql-action v3 → v4 | Current-main scan, permissions and runner compatibility |
| #4 | checkout v4 → v7 | All workflows on refreshed branch; checkout credentials/runner changes reviewed |
| #5 | setup-python v5 → v7 | Python 3.11–3.13 and cache behavior on supported runners |
| #6 | upload-pages-artifact v3 → v5 | Strict docs build and compatible deploy-pages artifact flow |
| #7 | pytesseract >=0.3.10 → >=0.3.13 | OCR-extra resolution and real fixture with Tesseract binary |
| #8 | xgboost >=2.0 → >=3.2.0 | Fit/predict/save/load through Aire adapter on declared platforms |
| #9 | pillow >=10.0 → >=12.3.0 | Vision/OCR decode tests and Python compatibility; minimum-floor justification |
| #10 | mkdocstrings >=0.25 → >=1.0.6 | Strict API docs build and Python compatibility; minimum-floor justification |
| #11 | keras >=3.0 → >=3.15.1 | Explicit backend configuration, train/save/load through Aire adapter |
| #12 | peft >=0.10 → >=0.20.0 | Compatible Transformers/Torch combination, tiny LoRA fit/save/resume |

Package PRs raise minimum supported versions; existing open-ended ranges already admit newer
versions. Treat floor changes separately from locking a tested version. Refresh and verify each
PR; close one as redundant only if its exact change is incorporated elsewhere and linked.
