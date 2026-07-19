# llm-bench

A comprehensive TUI for benchmarking OpenAI-compatible and Anthropic-compatible LLM endpoints.

## Features

- **Clean TUI** built with [Textual](https://textualize.io) — 6-screen workflow from config to results
- **Auto-discovers models** from `/v1/models` (OpenAI) or `/v1/messages` (Anthropic), with manual fallback
- **8 built-in benchmarks** covering:
  - Math reasoning (GSM8K-lite, 20 samples)
  - Code generation (HumanEval-lite, 15 samples, sandboxed execution)
  - Code explanation (LLM-as-judge, 10 samples)
  - Summarization (ROUGE-L F1, 10 samples)
  - Instruction following / JSON mode (12 samples, JSON Schema validation)
  - Long-context needle-in-haystack (5 context sizes)
  - Creative writing (LLM-as-judge, 8 samples)
  - Latency / throughput (TTFT, tokens/sec, p50/p95)
- **Custom benchmarks** via YAML/JSON
- **LLM-as-judge** with configurable rubrics; can reuse a benchmarked model or a separate endpoint
- **Conclusive recommendations**: "best for coding", "best for reasoning", "fastest", "cheapest", "best long-context", etc.
- **Exports**: JSON, Markdown, terminal table
- **Config persistence** in `~/.llm-bench.yaml` (0600 perms)

## Installation

```bash
pip install -e .[full]   # optional: tiktoken, rouge-score
```

Or from PyPI when published:

```bash
pip install llm-bench
```

## Quick Start

```bash
llm-bench
```

This launches the TUI. Navigate with:
- **Tab** / **Shift+Tab** — focus next/previous
- **Enter** — activate button / select row
- **Space** — toggle checkbox / select row in model table
- **/** — filter models
- **Ctrl+N** — next screen
- **Esc** — back / quit
- **P** — pause (during run)
- **C** — cancel (during run)

### Screens

1. **Endpoint** — Base URL, API key, protocol (auto / OpenAI / Anthropic). "Test connection" probes `/v1/models`.
2. **Models** — DataTable of discovered models with context window, modalities, features. Multi-select with Space.
3. **Benchmarks** — Toggle 8 built-in tasks + "Add custom" + "Advanced" (per-task sample count).
4. **Judge** — Pick judge: one of selected models / separate endpoint / skip LLM judge.
5. **Run** — Live per-task progress bars + streaming log. Pause/Resume/Cancel.
6. **Results** — Sortable table + recommendations chips. Export Markdown/JSON.

## Example: Custom Benchmark (YAML)

```yaml
# my_bench.yaml
name: "my_classification"
description: "Classify sentiment with JSON output"
samples:
  - id: s1
    prompt: |
      Classify: "I love this product!"
      Respond with JSON: {"sentiment": "positive|negative|neutral"}
    grader: json_schema
    schema:
      type: object
      properties:
        sentiment: {type: string, enum: [positive, negative, neutral]}
      required: [sentiment]
```

Load via **"Add custom benchmark"** in the TUI, or pass programmatically:

```python
from llm_bench.benchmarks.custom import CustomBenchmark, CustomSpec
bench = CustomBenchmark(CustomSpec.from_path("my_bench.yaml"))
```

## Programmatic Use

```python
import asyncio
from llm_bench.benchmarks import MathReasonBenchmark, CodeGenBenchmark
from llm_bench.clients import make_client
from llm_bench.runner import run_all
from llm_bench.config import PricingConfig, PricingEntry

async def main():
    client = make_client("openai", "https://api.openai.com/v1", "sk-...")
    result = await run_all(
        models=["gpt-4o-mini", "gpt-4o"],
        benchmarks=[MathReasonBenchmark(n_samples=5), CodeGenBenchmark(n_samples=3)],
        make_client=lambda m: make_client("openai", "https://api.openai.com/v1", "sk-..."),
        pricing=PricingConfig(entries={
            "gpt-4o-mini": PricingEntry(input_per_million=0.15, output_per_million=0.6),
            "gpt-4o": PricingEntry(input_per_million=2.5, output_per_million=10.0),
        }),
        judge_model="gpt-4o",  # for creative writing / code explanation
    )
    print(result.by_model_task)

asyncio.run(main())
```

## Benchmark Details

| Task | Samples | Grader | Notes |
|------|---------|--------|-------|
| `math_reasoning` | 20 | Numeric regex (`#### N` or last number) | GSM8K-style word problems |
| `code_generation` | 15 | Sandboxed Python subprocess | HumanEval-lite; 10s timeout, 256MB RAM |
| `code_explanation` | 10 | LLM judge (1-5 rubric) | Reference answer provided |
| `summarization` | 10 | ROUGE-L F1 | CNN/DM-style passages |
| `instruction_json` | 12 | JSON Schema validation | Extraction, classification, formatting |
| `long_context` | 5 sizes | Exact substring match | Needle at ~1k, 8k, 32k, 64k, 128k words |
| `creative_writing` | 8 | LLM judge (1-5) | Poems, stories, jokes, 6-word stories |
| `latency` | 10 | Throughput metrics | TTFT, median/p95 latency, tokens/sec |

### Recommendation Categories

| Category | Based on |
|----------|----------|
| `best_overall` | Mean pass-rate across all deterministic tasks |
| `best_coding` | `code_generation` + `code_explanation` |
| `best_reasoning` | `math_reasoning` + `instruction_json` |
| `best_long_context` | `long_context` pass rate at largest tested size |
| `best_json_mode` | `instruction_json` pass rate |
| `fastest` | Highest median tokens/sec (`latency`) |
| `cheapest` | Lowest cost across sampled tasks |
| `best_writing` | `creative_writing` + `code_explanation` judge scores |

## Configuration File

`~/.llm-bench.yaml` (created on first run):

```yaml
endpoint:
  name: default
  base_url: https://api.openai.com/v1
  api_key: "**********"
  protocol: auto
  judge_base_url: null
  judge_api_key: null
  judge_protocol: null
  judge_model: null
pricing:
  entries: {}
last_models:
  - gpt-4o-mini
  - gpt-4o
last_benchmarks:
  - math_reasoning
  - code_generation
  - summarization
sample_overrides: {}
```

Add per-model pricing under `pricing.entries` to enable cost estimates and "cheapest" recommendation.

## CLI Flags

```bash
llm-bench --help
```

Currently: `--no-tui` (reserved for future non-interactive mode).

## Requirements

- Python 3.11+
- `httpx` (installed)
- `textual` (installed)
- Optional: `tiktoken` (token counting), `rouge-score` (better ROUGE-L)

## License

MIT