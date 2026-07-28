"""Creative writing: LLM-judged on a 1-5 rubric."""

from __future__ import annotations

from pathlib import Path

from smoke_bench.benchmarks._io import load_json_or_jsonl
from smoke_bench.benchmarks.base import Benchmark, Sample

_DATA = Path(__file__).parent / "datasets" / "creative_writing.jsonl"


class CreativeWritingBenchmark(Benchmark):
    name = "creative_writing"
    description = "Short creative pieces; LLM-judge on a 1-5 rubric."

    def __init__(self, n_samples: int | None = None) -> None:
        super().__init__(n_samples)

    @property
    def samples(self) -> list[Sample]:
        out: list[Sample] = []
        for d in load_json_or_jsonl(_DATA):
            out.append(
                Sample(
                    id=d["id"],
                    prompt=d["prompt"],
                    grader=d.get("grader", "judge"),
                    rubric=d.get("rubric"),
                    request_kwargs={"max_tokens": 4096, "temperature": 0.7},
                )
            )
        return out
