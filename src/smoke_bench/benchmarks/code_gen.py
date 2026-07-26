"""HumanEval-lite: code generation with sandboxed test execution."""

from __future__ import annotations

from pathlib import Path

from smoke_bench.benchmarks._io import load_json_or_jsonl
from smoke_bench.benchmarks.base import Benchmark, Sample

_DATA = Path(__file__).parent / "datasets" / "humaneval_lite.jsonl"

_INSTRUCTION = (
    "Write a Python function that solves the following problem. "
    "Return only the function definition in a ```python fenced code block."
)


class CodeGenBenchmark(Benchmark):
    name = "code_generation"
    description = "HumanEval-lite: write a function, run hidden unit tests in a sandbox."

    def __init__(self, n_samples: int | None = None) -> None:
        super().__init__(n_samples)

    @property
    def samples(self) -> list[Sample]:
        out: list[Sample] = []
        for d in load_json_or_jsonl(_DATA):
            out.append(
                Sample(
                    id=d["id"],
                    prompt=_INSTRUCTION + "\n\n" + d["prompt"],
                    grader="sandbox",
                    test_code=d["test_code"],
                    request_kwargs={"max_tokens": 1024, "temperature": 0.0},
                )
            )
        return out
