# Aire capability status

The latest iteration is the [0.4.2 audit](docs/audits/2026-10-15-042-audit.md):
configuration secrecy/path checks, dataset snapshots/partitions, execution context
restoration, scheduler replacement, offline diagnostic smoke and repeatable
core/optional audit evidence. Run `make audit` and review its selected profile;
a green report does not certify unrun external services or model quality.

The remaining sections describe the historical 0.4.0 completion baseline.


The [master plan](docs/superpowers/plans/2026-10-14-aire-master-plan.md) implements
0.4.0 against the agreed scope: fulfill documented capabilities with reliable
packaging and honest limits. The historical [audit](docs/audits/2026-10-14-repository-audit.md)
retains the original failures; its original statuses are not current backlog items.

Use the [capability matrix](docs/capabilities.md) for the supported operations,
behavioral evidence and intentional limits. The [release QA record](docs/audits/2026-10-14-release-qa.md)
records the completed checks, merged PRs and public Pages deployment. The
[final review](docs/audits/2026-10-14-final-review.md) records findings and decisions.

Tasks 1–11 are implemented and locally verified, including packaging, streaming
safety, cancellation, remote stores, deployment, durable execution, evaluation,
optional CPU backends, MCP/tools and CLI examples. Three final-review defects
were reproduced and repaired: retries after cancellation, nested schema bounds,
and persistent cache identity after route changes. Task 12 completed exact-head
CI, dependency-PR disposition, merging and
main-only public Pages deployment. PRs #1–#12 and #15 are merged; audit tracker
#14 is closed. The final delivery record is linked above.

Intentional boundaries include the documented MCP/schema subset, labeled toy and
stub behavior, single-writer local persistence, at-least-once queue effects and
backend-specific transaction/query limits. Paid-provider/Pinecone live behavior,
hosted model quality, GPU quantization and production-scale throughput remain
unverified. A version bump is not a PyPI upload. No test/coverage percentage
establishes complete correctness or universal model quality.
