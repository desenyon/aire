# Capability completion design

## Agreed objective

Fulfill all documented capabilities, with reliable packaging and honest limits.
Preserve the vendor-neutral, offline-first Python library and the `AI` facade.
The deliverable is a trusted path from install to composition, evaluation and deployment.
“100%” means every supported requirement has an implementation and verification record;
it does not mean perfect coverage, no undiscovered bugs, or removing every intentional subset.

## Constraints

- Distribution: `aire-ai`; import: `aire`; CLI: `aire`.
- Python: preserve tested 3.11, 3.12, 3.13 support; establish macOS arm64 and Linux verification.
- Preserve frozen 0.3.x exports. Breaking removals require a minor release and migration notes.
- Core stays offline-capable with four runtime dependencies and lazy optional integrations.
- Existing public extras and provider-named no-op extras remain compatible during cleanup.
- No live paid provider calls, model downloads, service deployment, or package release are needed for the initial audit.
- Credentials never appear in fixtures, traces, locks, issue bodies, or examples.
- Real backend success requires real backend evidence; dry runs do not establish model quality.

## Approaches considered

1. **Repair contracts in place (selected).** Add failing behavioral tests, fix the boundaries,
   then extract repeated internals. Preserves public API and produces reviewable increments.
2. **Split into many distributions now.** Gives stronger optional-package isolation, but adds
   release coordination, migration and compatibility overhead before fixing current defects.
   Revisit only if independent release demand or measured dependency conflicts justify it.
3. **Rewrite the facade and replace experimental subsystems.** Largest migration risk and
   weakest short-term verification. Does not address the known failures efficiently.

## Contract architecture

Keep `Runtime` as composition root. Use one execution-control path for normal/event/token
agent execution, one cache eligibility/fingerprint/replay contract, and shared HTTP request
validation/output-policy helpers. Each remote store must implement stable identity, status
handling, pagination and explicit replacement capabilities. Runtime ownership determines
shutdown responsibilities; generated apps fail closed for explicitly configured backends.

Use a capability matrix that maps each public namespace to supported operations, limitations,
source, test evidence, and maturity. Core bugs cannot be closed merely by relabeling them
experimental. Features outside product non-goals remain excluded: hosted infrastructure,
full MCP ecosystem, pretrained foundation weights bundled with Aire, replacing ML frameworks.

## Packaging direction

Keep `pyproject.toml` as published metadata; retain bounded core and optional dependency
contracts. Add a tested development lock and separate test/docs/release dependency groups.
Use a single version source. Generate exact deployment constraints for a chosen target
platform, distinct from `aire.lock` (which stores model/store references).

Do not assume the heavy `all` extra is universally resolvable. Test supported profiles and
make platform exclusions explicit. Remove duplicated constraint maintenance only after
wheel metadata and extras-install equivalence tests protect compatibility.

## Completion gates

Every supported capability must have a happy-path test, meaningful failure-path tests,
a runnable example, accurate describe/docs output, installation-profile evidence, and a
reviewed limitation statement. Prioritize all P0/P1 audit findings. For optional external
adapters, deterministic transport contracts plus a recorded live service test are required
before calling them verified. Block release on missing required evidence.

Known gaps and repair criteria: [audit](../../audits/2026-10-14-repository-audit.md).
