"""Latency benchmark: time-to-first-token, total latency, output tokens/sec."""

from __future__ import annotations

from typing import Sequence

from llm_bench.benchmarks.base import Benchmark, Sample

_PROMPTS = [
    "Count from 1 to 20, one number per line.",
    "List the 7 days of the week, one per line.",
    "Name 5 capital cities, one per line.",
    "Name 4 primary colors, one per line.",
    "List 6 fruits, one per line.",
    "Name the 9 planets of the solar system, one per line.",
    "List 8 European countries, one per line.",
    "Count from 10 down to 1, one number per line.",
    "Name 5 programming languages, one per line.",
    "List 6 common animals, one per line.",
]


class LatencyBenchmark(Benchmark):
    name = "latency"
    description = "TTFT, total latency, output tokens/sec over fixed short prompts."

    def __init__(self, n_samples: int | None = None) -> None:
        super().__init__(n_samples)
        self._all_samples: list[Sample] = [
            Sample(
                id=f"lat_{i+1}",
                prompt=p,
                grader="contains",
                expected="1",
                request_kwargs={"max_tokens": 128, "temperature": 0.0},
            )
            for i, p in enumerate(_PROMPTS)
        ]

    @property
    def samples(self) -> Sequence[Sample]:
        if self.n_samples_override is not None:
            return self._all_samples[: self.n_samples_override]
        return self._all_samples
