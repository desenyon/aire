# Contributing

Development setup, coding standards, and PR expectations live in the repository root:

→ **[CONTRIBUTING.md](https://github.com/desenyon/aire/blob/main/CONTRIBUTING.md)**

Quick local loop:

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Be honest in docs and `.describe()` when adding experimental APIs — see [Honesty](honesty.md).
