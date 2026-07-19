# llm-bench

Comprehensive TUI for benchmarking OpenAI- and Anthropic-compatible LLM endpoints.

## Quick Start

```bash
# 1. Create & activate virtual environment
make venv
source .venv/bin/activate

# 2. Launch the TUI
llm-bench
```

## TUI Walkthrough

| Screen | Purpose | Keys |
|--------|---------|------|
| **1. Endpoint** | Base URL, API key, protocol (auto/openai/anthropic) | `Ctrl+T` test, `Ctrl+N` next |
| **2. Models** | Auto-fetched list with context/modalities/features; multi-select | `Space` toggle, `A` all, `N` none, `/` filter |
| **3. Benchmarks** | Toggle 8 built-in tasks; add custom YAML; "Advanced" for per-task sample counts | `Ctrl+N` next |
| **4. Judge** | Pick judge model: from selected / separate endpoint / skip | `Ctrl+N` next |
| **5. Run** | Live progress bars + streaming log; pause/resume/cancel | `P` pause, `C` cancel |
| **6. Results** | Sortable table + recommendation chips; export JSON/MD | `E` export, `Esc` new run |

## Built-in Benchmarks (8)

| Task | Samples | Grader | Measures |
|------|---------|--------|----------|
| Math / Reasoning (GSM8K-lite) | 20 | Regex → numeric match | Accuracy |
| Code Generation (HumanEval-lite) | 15 | Sandboxed subprocess exec | Pass@1 |
| Code Explanation | 10 | LLM-judge rubric 1–5 | Quality |
| Summarization | 10 | ROUGE-L F1 | Fidelity |
| Instruction / JSON Mode | 12 | JSON Schema validation | Schema compliance |
| Long Context (Needle-in-haystack) | 5 sizes (1k–128k) | Exact substring | Effective context |
| Creative Writing | 8 | LLM-judge rubric 1–5 | Creativity |
| Latency / Throughput | 10 | Streaming metrics | TTFT, tokens/sec, p50/p95 |

## Custom Benchmarks

Create `my_task.yaml`:

```yaml
name: my_custom_task
description: "Extract structured data"
samples:
  - id: s1
    prompt: "Parse: 'John, 30, NYC' → JSON with name, age, city"
    grader: json_schema
    schema:
      type: object
      properties:
        name: {type: string}
        age: {type: integer}
        city: {type: string}
      required: [name, age, city]
```

In the TUI: **Benchmarks → Add custom benchmark** → paste YAML.

## Recommendations Engine

After a run, the **Results** screen shows chips for:

- `best_overall` — mean pass-rate across deterministic tasks
- `best_coding` — code_gen + code_explain
- `best_reasoning` — math + instruction_json
- `best_long_context` — needle retrieval at max tested size
- `best_json_mode` — instruction_json pass-rate
- `best_writing` — creative_writing + code_explain (judge score)
- `fastest` — median tokens/sec (latency task)
- `cheapest` — lowest USD cost (requires pricing config)

Tie-breakers: lower latency → lower cost.

## Pricing (Optional)

Add per-model pricing in `~/.llm-bench.yaml` (auto-created):

```yaml
pricing:
  entries:
    gpt-4o:
      input_per_million: 5.0
      output_per_million: 15.0
    gpt-4o-mini:
      input_per_million: 0.15
      output_per_million: 0.60
```

## Output Artifacts

Each run creates `./llm_bench_results/run_YYYYMMDD_HHMMSS/`:

```
run_20260719_143022/
├── summary.json      # models, tasks, errors
├── details.json      # per-task per-model scores
├── samples.jsonl     # one line per sample (streamable)
├── report.md         # human-readable markdown
```

## Development

```bash
make venv          # create .venv + install
make test          # pytest (38 tests)
make lint          # ruff
make fmt           # ruff --fix
make run           # launch TUI
```

## Requirements

- Python 3.11+
- `httpx`, `textual`, `pydantic`, `pyyaml`, `jsonschema`, `rich`
- Optional: `tiktoken`, `rouge-score` (auto-fallback if missing)

## License

MIT