"""Persist run results to disk (JSONL append + final JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from llm_bench.runner import RunResult


def _sample_dict(d):  # type: ignore[no-untyped-def]
    if hasattr(d, "__dataclass_fields__"):
        return asdict(d)
    return d


def save_result(result: RunResult, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Summary
    summary = {
        "models": sorted({m for m, _ in result.by_model_task.keys()}),
        "tasks": sorted({t for _, t in result.by_model_task.keys()}),
        "errors": [list(e) for e in result.errors],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-task details
    details: dict[str, dict[str, dict]] = {}
    for (model, task), score in result.by_model_task.items():
        details.setdefault(task, {})[model] = {
            "n": score.n,
            "passed": score.passed,
            "mean_score": score.mean_score,
            "pass_rate": score.pass_rate,
            "median_latency_s": score.median_latency_s,
            "p95_latency_s": score.p95_latency_s,
            "mean_tokens_per_s": score.mean_tokens_per_s,
            "total_input_tokens": score.total_input_tokens,
            "total_output_tokens": score.total_output_tokens,
            "cost_usd": score.cost_usd,
            "per_sample": [_sample_dict(s) for s in score.per_sample],
        }
    (run_dir / "details.json").write_text(json.dumps(details, indent=2))

    # JSONL append for streaming consumers
    with (run_dir / "samples.jsonl").open("w") as f:
        for (model, task), score in result.by_model_task.items():
            for s in score.per_sample:
                f.write(
                    json.dumps(
                        {"model": model, "task": task, **_sample_dict(s)}
                    )
                    + "\n"
                )

    return run_dir
