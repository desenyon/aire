# 0.4.0 final review and decisions

This historical review covers the implementation merge. The subsequent request
to publish is tracked separately in the [publication repair](2026-10-14-release-publication.md).

A fresh reviewer inspected baseline 70b4633 through e3e96de, ran 78 focused
boundary tests, and reproduced three Important defects. There were no confirmed
Critical findings. All three were accepted and repaired in 89e2dcc with regressions
observed failing before implementation, then passing. The complete suite passed
308 tests, with 11 live tests selected separately and four non-failing warnings.

| Finding | Repair / regression |
| --- | --- |
| Cancellation/deadline allowed further tool retries | Check effective context before every attempt; four execution-control cases |
| Schema constraints disappeared in arrays and unions | Compile constrained annotations recursively; six valid/invalid schema pairs |
| Persistent cache survived incompatible route changes | Fingerprint route order/policy and embedder identity; explicit namespace; reject stale resolved refs; six cache cases |

The reviewer noted stale release-QA evidence as Minor. It was regraded Important
because reconciling the final evidence is an explicit delivery gate, and refreshed
with executed results. No deferred minors remain. External-service, hardware and
scale boundaries were evaluated explicitly below; they are not implicit success
claims. No second review was used to replace the regression/full-suite gate.

## Rulings and costs

- Ruling: Continue in the existing clean dedicated PR branch; no second worktree needed for isolation from main. Cost if wrong: branch checkout needed; no user changes present.
- Ruling: Target 0.4.0, preserve frozen exports, integrate dependency PRs only after refreshed backend/workflow QA. Cost if wrong: release-version adjustment.
- Ruling: Install local QA tools (Colima/Docker, Redis, Tesseract, FFmpeg) because the requested full backend/deployment verification lacks a preexisting runtime. Cost if wrong: removable tools and local VM disk usage.
- Ruling: Add skops to ML extra for safe sklearn persistence; no pickle auto-loading or trusting arbitrary serialized types. Cost if wrong: extra optional dependency and format migration.
- Ruling: Optional CPU QA uses OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Unbounded combined native runtime execution crashed; bounded runs pass. Cost if wrong: unconstrained multithread performance is unverified; isolated profile CI remains required.
- Task 12: Ruling: mkdocstrings 1.0.6 fails with pymdown-extensions 12.3 (Highlighter lacks md argument); raise the reviewed floor to 1.0.7, which passes strict docs. Cost if wrong: older docs tooling no longer supported.
- Task 11: Ruling: the plan's count of 24 manifest namespaces was a typo; baseline70b4633 and the frozen facade contain23. Preserve those23 and document project/UI entrypoints separately. Cost if wrong: an intended new namespace would require a separate additive API decision.
- Task 12: Ruling: GitHub Pages environment permits only main; branch deployment was rejected before steps ran. Preserve that protection, finish code QA/review before merge, then verify deployment from main. Cost if wrong: a Pages workflow repair may be needed after merge; no protection is bypassed.
- Final: Ruling: paid providers, hosted media quality and Pinecone remain transport-tested only — no live credentials or paid execution; no live-quality claim — cost if wrong: provider-specific failures remain undiscovered.
- Final: Ruling: GPU/bitsandbytes remains explicitly unverified — available QA hardware is CPU — cost if wrong: GPU correctness/performance defects remain undiscovered.
- Final: Ruling: pretrained/model quality remains caller-owned; toy architectures are labeled — finite toy training proves execution, not universal quality — cost if wrong: callers need additional quality evaluation.
- Final: Ruling: distributed transactions/multiwriter files/exactly-once queue effects remain outside contract — documented nontransactional/single-writer/at-least-once semantics — cost if wrong: users needing stronger guarantees must add coordination/idempotency.
- Final: Ruling: running synchronous thread bodies cannot be forcibly stopped — retries will honor cancellation, while hard termination requires processes — cost if wrong: an already-running side effect can finish after timeout.
- Final: Ruling: full MCP hosting/event resumption/unsupported compositions remain explicit subsets — supported client/schema behavior is tested — cost if wrong: additional protocol/schema integration work is required.
- Final: Ruling: production-scale remote retrieval is not proven — bounded live fixtures verify correctness within documented backend/query limits — cost if wrong: large workloads need backend-specific performance/limit validation.
- Final: Ruling: merge and main-only Pages verification finish this authorized delivery; PyPI upload remains unperformed — publishing is not needed to prove versioned artifacts and was not requested — cost if wrong: package-index availability requires a separate release tag/publication.
- Final: Ruling: wheel/hash and optional compatibility claims are scoped to executed platforms/profiles — installed artifact/hash checks do not prove every future resolver combination — cost if wrong: unsupported combinations need additional matrix coverage.

## Completion

All twelve plan tasks are complete. Task 12's final merged-tree run passed
308 tests (11 live tests selected separately); merged-main CI and the protected
Pages deployment also passed. The release QA record retains exact run/deployment
identities and individual dependency-PR outcomes. The temporary SDD ledger is
retired after preserving its decisions here; sibling workspaces are untouched.
