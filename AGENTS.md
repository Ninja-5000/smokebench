# AGENTS.md

## Project

`smokebench` — a Textual TUI for smoke-testing OpenAI/Anthropic-compatible LLM endpoints. Python 3.11+, src-layout package (`src/smoke_bench`). Entry point: `smoke_bench.app.run()` → `LLMBenchApp`; the `--no-tui` flag is reserved and unused.

## Dev commands

```sh
make venv        # create .venv + install -e ".[dev,full]" (tiktoken, rouge-score optional)
make test        # pytest -q (offline; all HTTP mocked via respx, TUI via Textual pilot)
make lint        # ruff check src/ tests/
make fmt         # ruff check --fix src/ tests/
make run         # smokebench — TUI, needs a live endpoint (default http://127.0.0.1:1234/v1)
```

- `make typecheck` silently no-ops: mypy is not installed/configured — don't treat it as a gate.
- Ruff: line-length 100, E501 ignored; select E/F/W/I/B/UP.
- Test suite is fully offline and fast; never point tests at a real endpoint.

## Runtime config / secrets

- `smokebench.yaml` is auto-created at the repo root on first run, gitignored, chmod 0600, and contains real API keys (one exists locally). Never commit it; it's already covered by `.gitignore`.
- `SMOKEBENCH_CONFIG` env var overrides the config path (default `smokebench.yaml`).
- `smokebench_results/` (run artifacts) is gitignored runtime output.

## Gotchas

- Version lives in two places that have drifted: `pyproject.toml` `version = 1.1.0` vs `src/smoke_bench/__init__.py` `__version__ = "1.0.0"`. Keep them in sync when releasing.
- Releases: pushing a `v*` tag runs `.github/workflows/release.yml` (build → verify package data in wheel → GitHub release → PyPI). Release notes are pulled from the matching `## [version]` section in `CHANGELOG.md`.
- `styles.tcss` and `benchmarks/datasets/*.jsonl` must stay listed under `[tool.setuptools.package-data]` in `pyproject.toml`; CI fails if they're missing from the wheel.

## Architecture

- `clients/` — OpenAI/Anthropic compat clients with auto protocol detection (`clients/detect.py`).
- `benchmarks/` — 8 benchmark classes + lite dataset JSONLs (math, code_gen, code_explain, summarization, instruction_json, long_context, creative_writing, latency). Custom YAML benchmarks load at runtime.
- `runner.py` — two-phase pipeline: Phase 1 collects model outputs concurrently (bounded by `max_concurrency`), Phase 2 grades serially so the judge never receives overlapping requests. Latency comes from the API-reported `latency_s`, not wall-clock.
- `judging/sandbox.py` — code_gen grader executes model code in a `python -I -B` subprocess with rlimits; POSIX-only (`preexec_fn`).
- `tui/` — screens + `styles.tcss`; `results/` writes `summary.json` / `details.json` / `samples.jsonl` / `report.md`.