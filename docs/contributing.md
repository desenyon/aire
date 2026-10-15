# Contributing

Development setup, coding standards, and PR expectations live in the repository root:

→ **[CONTRIBUTING.md](https://github.com/desenyon/aire/blob/main/CONTRIBUTING.md)**

Quick local loop:

```bash
uv sync --locked --no-default-groups --group dev
make audit
```

Be honest in docs and `.describe()` when adding experimental APIs — see [Honesty](honesty.md).

## Dependency and workflow QA

`pyproject.toml` defines published requirements; `uv.lock` records the tested
resolution. Use `uv sync --locked --no-default-groups --group dev` and selected extras for optional
backend work. Dependabot uses the [uv ecosystem](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
to update metadata and the lock together. Workflow/tooling updates are grouped;
ML and service backend updates still require their relevant behavioral checks.

Core CI tests Linux/macOS and Python 3.11–3.13, plus the oldest available binary
versions of direct core/serve requirements. Optional profiles exercise actual
CPU backends and reviewed version floors. `scripts/verify_services.py --wheel
<path>` creates disposable local Docker services, runs Aire CRUD/recovery
contracts and builds a hash-locked app with that wheel. It verifies requests,
cache replay, backend outages/recovery, failed startup and graceful shutdown,
then removes only resources it created. Install Docker and uv separately.

The benchmark guard compares warmed medians with a repeatable committed baseline,
a 5x ratio and a 1ms absolute noise floor. It catches material regressions and
missing/nonfinite measurements; it is not a cross-machine throughput guarantee.

Core CI retains source-bound JSON audit reports and per-check logs. See the
[0.4.2 audit](audits/2026-10-15-042-audit.md) for profiles and interpretation.
