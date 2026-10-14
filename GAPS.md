# Aire capability backlog

The [2026-10-14 repository audit](docs/audits/2026-10-14-repository-audit.md)
supersedes the previous claim that all prioritized depth items were closed.
Core offline checks pass, but reproduced failures and unverified integrations remain.

The [master plan](docs/superpowers/plans/2026-10-14-aire-master-plan.md) defines the
repair order, affected files, regression tests and completion gates.
Scope: fulfill documented capabilities with reliable packaging and honest limits.

| Area | Current status | Plan |
| --- | --- | --- |
| Distribution/install names | Repaired in source; release pending | Task 1 |
| Reproducible dependencies, extras, wheel/deployment validation | Open | Task 2 |
| Streaming output guardrails and cache isolation | Reproduced failures | Task 3 |
| Agent streaming budgets, cancellation, synchronous tool timeouts | Reproduced failures | Task 4 |
| Remote store IDs, document identity and HTTP failures | Reproduced Qdrant failures; adapter-wide verification open | Task 5 |
| Deployment startup/readiness/lifecycle | Configured-backend fallback needs repair | Task 6 |
| Workflows, queues, sessions, graph, memory, scheduler | Local recovery baseline passes; broader contracts open | Task 7 |
| Evaluation, analytics and telemetry | Analytics mislabel reproduced; wider verification open | Task 8 |
| ML/training/media/OCR | Optional profiles not comprehensively verified | Task 9 |
| MCP/plugins/tools | Offline subset tests pass; transport and schema contracts open | Task 10 |
| CLI/examples/capability docs | Examples exit successfully; behavioral assertions/docs reconciliation open | Task 11 |
| GitHub dependency PRs, Pages, release gates | PRs #1–#12 need fresh checks; Pages deployment failed | Task 12 |

Intentional boundaries remain: a library rather than a hosted platform; MCP subset;
explicit toy/dry-run/stub behavior; local-only file queue; at-least-once recovery;
backend-specific transaction limits. See [product scope](docs/product.md) and
[honesty notes](docs/honesty.md). These boundaries do not excuse incorrect results
inside an advertised supported operation.

A row is complete only when its tests, examples, docs and release evidence agree.
No percentage of test coverage establishes full capability completion.
