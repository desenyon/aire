.PHONY: install lint typecheck test docs bench release-check all

UV_RUN = uv run --locked --no-default-groups --group dev

install:
	uv sync --locked --no-default-groups --group dev

lint:
	$(UV_RUN) ruff check src tests scripts
	$(UV_RUN) ruff format --check src tests scripts

typecheck:
	$(UV_RUN) mypy

test:
	$(UV_RUN) pytest -m "not live and not optional"

docs:
	$(UV_RUN) mkdocs build --strict

bench:
	$(UV_RUN) python benchmarks/microbench.py --json benchmarks/results/latest.json

release-check:
	$(UV_RUN) python scripts/release.py check

all: lint typecheck test docs release-check
