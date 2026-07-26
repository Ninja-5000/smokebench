"""Tests for the recommendation engine."""

from __future__ import annotations

from smoke_bench.benchmarks.base import SampleResult, TaskScore
from smoke_bench.results.recommend import recommend
from smoke_bench.runner import RunResult


def _ts(task: str, n: int = 5, passed: int = 5, mean_score: float = 1.0, tps: float = 30.0, cost: float = 0.01) -> TaskScore:
    return TaskScore(
        task=task,
        n=n,
        passed=passed,
        mean_score=mean_score,
        median_latency_s=1.0,
        p95_latency_s=1.5,
        mean_tokens_per_s=tps,
        total_input_tokens=1000,
        total_output_tokens=500,
        cost_usd=cost,
        per_sample=[
            SampleResult(
                sample_id=f"{task}_{i}",
                output="",
                score=1.0 if i < passed else 0.0,
                passed=i < passed,
                latency_s=1.0,
                ttft_s=0.1,
                input_tokens=200,
                output_tokens=100,
            )
            for i in range(n)
        ],
    )


def test_recommend_picks_top_models() -> None:
    r = RunResult()
    r.by_model_task[("m1", "math_reasoning")] = _ts("math_reasoning", passed=5, tps=20, cost=0.02)
    r.by_model_task[("m2", "math_reasoning")] = _ts("math_reasoning", passed=3, tps=40, cost=0.01)
    r.by_model_task[("m1", "code_generation")] = _ts("code_generation", passed=4, tps=25, cost=0.05)
    r.by_model_task[("m2", "code_generation")] = _ts("code_generation", passed=5, tps=30, cost=0.04)
    r.by_model_task[("m1", "latency")] = _ts("latency", passed=5, tps=20, cost=0.0)
    r.by_model_task[("m2", "latency")] = _ts("latency", passed=5, tps=80, cost=0.0)

    recs = {r.category: r for r in recommend(r)}
    # m2 should win "best_coding" (avg 4/5 + 5/5 = 0.9 vs m1 0.8+0.8=0.8)
    assert recs["best_coding"].model == "m2"
    # m2 should be "fastest"
    assert recs["fastest"].model == "m2"


def test_recommend_handles_missing_data() -> None:
    r = RunResult()
    recs = recommend(r)
    # All categories should still return, possibly with model=None.
    assert all(rec.category for rec in recs)
