# Contributing to aire

Thanks for contributing. aire is an **agent-first**, **offline-capable** Python library
(Apache-2.0). Changes that preserve discoverability (`.describe()`), provider
independence, and honest stub behavior are preferred.

## Development setup

Requires **Python 3.11+**.

```bash
git clone https://github.com/desenyon/aire.git
cd aire
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install               # pip install -e ".[dev]"
pre-commit install         # optional but recommended
```

## Workflow

1. Open an issue for non-trivial changes (or discuss on an existing one).
2. Create a branch from `main`.
3. Implement with focused commits; prefer offline tests (`mock:echo`, `local:hashing`).
4. Run checks before opening a PR:

```bash
make lint
make typecheck
make test
# or
make all
```

5. Open a PR using the template. Link related issues.

## Code style

- Format and lint with **Ruff** (see `[tool.ruff]` in `pyproject.toml`).
- Typecheck with **mypy** (`strict` for `src/aire`).
- Public APIs should be typed and, where practical, expose `.describe()`.
- Do not add vendor SDKs to core; providers use HTTP adapters (`httpx`).
- Prefer honesty over polish for experimental surfaces (vision/audio/foundation):
  mark stubs clearly rather than overselling.

## Tests

Tests live under `tests/`. Markers include `integration`, `live`, `performance`,
and `security`. Live/network tests should be skipped by default.

```bash
pytest -q
pytest -m "not live"
```

Keep new tests deterministic and offline when possible.

## Docs

User-facing docs live in [`docs/`](docs/). Update them when you change public
behavior. Release notes go in [`CHANGELOG.md`](CHANGELOG.md).

## Security

Do not file public issues for vulnerabilities. See [`SECURITY.md`](SECURITY.md).

## Code of conduct

Be kind. We follow the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0 (see [`LICENSE`](LICENSE)).
