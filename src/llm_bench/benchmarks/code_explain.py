"""Code explanation benchmark. LLM-judged."""

from __future__ import annotations

from pathlib import Path

from llm_bench.benchmarks._io import load_json_or_jsonl
from llm_bench.benchmarks.base import Benchmark, Sample

_DATA = Path(__file__).parent / "datasets" / "code_explain.jsonl"


class CodeExplainBenchmark(Benchmark):
    name = "code_explanation"
    description = "Explain a code snippet. LLM-as-judge on a 1-5 rubric."

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
                    reference=d.get("reference"),
                    rubric=d.get("rubric"),
                    request_kwargs={"max_tokens": 512, "temperature": 0.0},
                )
            )
        return out
