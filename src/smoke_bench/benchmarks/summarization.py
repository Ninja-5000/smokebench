"""Summarization: ROUGE-L F1 against a reference summary."""

from __future__ import annotations

from pathlib import Path

from smoke_bench.benchmarks._io import load_json_or_jsonl
from smoke_bench.benchmarks.base import Benchmark, Sample

_DATA = Path(__file__).parent / "datasets" / "summarization.jsonl"

_INSTRUCTION = (
    "Summarize the following article in 2-3 sentences. Be concise and factual.\n\n"
)


class SummarizationBenchmark(Benchmark):
    name = "summarization"
    description = "Summarize a passage; graded by ROUGE-L F1 against a reference."

    def __init__(self, n_samples: int | None = None) -> None:
        super().__init__(n_samples)

    @property
    def samples(self) -> list[Sample]:
        out: list[Sample] = []
        for d in load_json_or_jsonl(_DATA):
            out.append(
                Sample(
                    id=d["id"],
                    prompt=_INSTRUCTION + d["source"],
                    expected=d["reference"],
                    grader="rouge_l",
                    request_kwargs={"max_tokens": 256, "temperature": 0.0},
                )
            )
        return out
