.PHONY: install lint typecheck test docs bench all

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	pytest -q -m "not live"

docs:
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs build --strict; \
	else \
		echo "install docs extras: pip install -e '.[docs]'"; \
		ls docs >/dev/null; \
	fi

bench:
	python benchmarks/microbench.py --json benchmarks/results/latest.json

all: lint typecheck test
