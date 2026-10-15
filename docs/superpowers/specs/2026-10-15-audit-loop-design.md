# 0.4.2 audit and diagnostics design

The user authorized continuing the audit/fix/feature loop through QA, PR merge
and publication in 0.4.2. The objective remains documented capability correctness,
reliable packaging and honest limits. This iteration fixes reproducible failures
and makes audits repeatable; it does not claim universal correctness.

## Contracts

- Missing explicitly requested configuration files/directories fail with
  `ConfigurationError`; implicit project/user/default discovery stays supported.
- CLI config inspection masks credentials, gateway bearer tokens and nested
  secret-bearing keys, including custom observability mask fields. It rejects
  unknown inspection selectors instead of printing an empty success object.
- Dataset record/metadata/lineage snapshots cannot be mutated through constructor
  inputs, accessors or transform callbacks. Sampling accepts finite fractions in
  [0, 1], with zero selecting zero records, and nonnegative integer counts;
  specifying both is an error. Split fractions must be finite, nonnegative and
  sum to one; partitions conserve records and agree with their manifest.
- Re-entering one execution context restores the prior ambient context correctly,
  including overlapping asyncio tasks. Schedule replacement discards stale
  handlers and failed registrations do not change working jobs; intervals and
  poll periods must be finite and positive.
- `aire doctor --smoke --output PATH` adds a credential-free generation,
  embedding, tool and cited RAG check. The versioned JSON report records pass,
  fail and skipped checks, version and Python; invalid config produces a failed
  report rather than a traceback. Default doctor remains offline. Live probes
  use resolved configuration credentials and close resources. No secret values
  are included in reports. Missing unrelated credentials are skipped.
- `scripts/audit.py --profile core|optional --output PATH` runs named repository
  checks with bounded subprocesses, individual logs and incrementally written
  JSON evidence. Core includes lint, format, types, core tests, behavioral
  examples, strict docs, release metadata, build and installed-wheel smoke.
  Optional runs the complete non-live suite in an already provisioned environment.
  Reports identify source commit, dirty state and content fingerprint; source
  changes during a run invalidate its result. Failed, timed-out and unrun checks
  never count as success. Reports live outside source or in ignored directories.
  CI runs the same core profile and preserves its evidence even on failure.
- Keep Python >=3.11, the four existing runtime dependencies and all public extras.
  No new runtime dependency. Retain lazy imports, release integrity gates and
  digital attestations. QA optional CPU backends with bounded threads.

## Alternatives and limits

A public diagnostics command helps installed users; a repository audit runner
helps contributors and CI. They share the reporting principles but have separate
responsibilities. A feature registry promising automatic completeness would
misrepresent what tests establish, so the capability matrix remains evidence
with explicit limits. External services, hosted quality and production scaling
remain separate live QA; a green core report does not certify them.

## Acceptance

Regression tests first reproduce each failure; each fix passes those tests.
Doctor smoke proves an offline outcome and produces machine-readable failure
reports. The audit runner is tested with real passing, failing, timing-out and
source-mutating subprocesses. Run core and optional audits, fresh distribution
installation, dependency audit, workflow lint and review before merging. Publish
only after exact-main CI/CodeQL/Docs/Benchmarks and release dry-run success, then
verify PyPI/GitHub artifact bytes and a fresh public-index installation.
