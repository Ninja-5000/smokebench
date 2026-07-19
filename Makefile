.PHONY: help install test lint fmt typecheck run clean venv

# Default: show help
help:
	@echo "llm-bench — Make targets"
	@echo ""
	@echo "  make install      Install package + dev deps into current venv"
	@echo "  make venv         Create .venv and install everything"
	@echo "  make test         Run pytest"
	@echo "  make lint         Run ruff"
	@echo "  make fmt          Run ruff --fix"
	@echo "  make typecheck    Run mypy (if configured)"
	@echo "  make run          Launch the TUI (llm-bench)"
	@echo "  make clean        Remove build artifacts + .venv"
	@echo ""

venv:
	@bash ./create-venv.sh

install:
	pip install -e ".[dev,full]"

test:
	pytest -q

lint:
	ruff check src/ tests/

fmt:
	ruff check --fix src/ tests/

typecheck:
	mypy src/llm_bench --ignore-missing-imports 2>/dev/null || echo "mypy not configured; skipping"

run:
	llm-bench

clean:
	rm -rf .venv build dist *.egg-info .pytest_cache .ruff_cache __pycache__ src/llm_bench.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true