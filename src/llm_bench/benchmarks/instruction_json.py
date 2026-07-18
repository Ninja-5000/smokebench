"""Instruction-following with JSON schema validation."""

from __future__ import annotations

from pathlib import Path

from llm_bench.benchmarks._io import load_json_or_jsonl
from llm_bench.benchmarks.base import Benchmark, Sample

_DATA = Path(__file__).parent / "datasets" / "instruction_json.jsonl"


class InstructionJSONBenchmark(Benchmark):
    name = "instruction_json"
    description = "Follow structured instructions and emit JSON that validates against a schema."

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
                    grader=d.get("grader", "json_schema"),
                    schema=d.get("schema"),
                    request_kwargs={"max_tokens": 512, "temperature": 0.0},
                )
            )
        return out
