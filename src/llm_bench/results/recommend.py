"""Recommend the best model for each task category from a ``RunResult``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from llm_bench.runner import RunResult


@dataclass
class Recommendation:
    category: str
    model: str | None
    score: float
    detail: str = ""


CATEGORIES: dict[str, dict] = {
    "best_overall": {
        "tasks": [
            "math_reasoning",
            "code_generation",
            "summarization",
            "instruction_json",
            "long_context",
        ],
        "metric": "pass_rate",
    },
    "best_coding": {
        "tasks": ["code_generation", "code_explanation"],
        "metric": "pass_rate",
    },
    "best_reasoning": {
        "tasks": ["math_reasoning", "instruction_json"],
        "metric": "pass_rate",
    },
    "best_long_context": {
        "tasks": ["long_context"],
        "metric": "pass_rate",
    },
    "best_json_mode": {
        "tasks": ["instruction_json"],
        "metric": "pass_rate",
    },
    "best_writing": {
        "tasks": ["creative_writing", "code_explanation"],
        "metric": "mean_score",
    },
    "fastest": {
        "tasks": ["latency"],
        "metric": "tokens_per_s",
    },
    "cheapest": {
        "tasks": ["math_reasoning", "code_generation", "summarization"],
        "metric": "cost",
    },
}


def _score_for(score, metric: str) -> float:
    if metric == "pass_rate":
        return score.pass_rate
    if metric == "mean_score":
        return score.mean_score
    if metric == "tokens_per_s":
        return score.mean_tokens_per_s
    if metric == "cost":
        return -score.cost_usd  # cheaper is better
    return 0.0


def recommend(result: RunResult) -> list[Recommendation]:
    out: list[Recommendation] = []
    for cat, cfg in CATEGORIES.items():
        tasks = cfg["tasks"]
        metric = cfg["metric"]
        per_model: dict[str, list[float]] = defaultdict(list)
        for (model, task), score in result.by_model_task.items():
            if task in tasks:
                per_model[model].append(_score_for(score, metric))
        if not per_model:
            out.append(Recommendation(category=cat, model=None, score=0.0, detail="no data"))
            continue
        ranked = sorted(
            per_model.items(),
            key=lambda kv: sum(kv[1]) / len(kv[1]),
            reverse=True,
        )
        model, vals = ranked[0]
        agg = sum(vals) / len(vals)
        detail = ", ".join(f"{v:.3f}" for v in vals)
        out.append(
            Recommendation(
                category=cat,
                model=model,
                score=agg,
                detail=f"({detail}) across {','.join(tasks)}",
            )
        )
    return out
