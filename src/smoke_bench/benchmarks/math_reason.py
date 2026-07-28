"""GSM8K-lite: numeric word problems. 20 samples, regex grader extracts the
final number from ``#### N`` or the last number in the response."""

from __future__ import annotations

import re
from pathlib import Path

from smoke_bench.benchmarks._io import load_json_or_jsonl
from smoke_bench.benchmarks.base import Benchmark, Sample
from smoke_bench.judging.deterministic import GradeResult

_DATA = Path(__file__).parent / "datasets" / "gsm8k_lite.jsonl"

_INSTRUCTION = (
    "Solve the following word problem. Show your reasoning, then end with "
    "the line '#### <number>' where <number> is the final numeric answer."
)

# Either "#### N" or the last number on its own line at end of output.
_ANSWER_RE = re.compile(
    r"(?:####\s*([\-+]?\d+(?:\.\d+)?))|(?:^|\n)\s*([\-+]?\d+(?:\.\d+)?)\s*\.?\s*$",
    re.MULTILINE,
)


def _grade_math(output: str, expected: str) -> GradeResult:
    matches = _ANSWER_RE.findall(output)
    if not matches:
        return GradeResult(score=0.0, passed=False, detail="no number found")
    last = matches[-1]
    extracted = next((m for m in last if m), None)
    if not extracted:
        return GradeResult(score=0.0, passed=False, detail="no number found")
    try:
        if abs(float(extracted) - float(expected)) < 1e-3:
            return GradeResult(score=1.0, passed=True, detail=extracted)
    except ValueError:
        pass
    return GradeResult(score=0.0, passed=False, detail=extracted)


class MathReasonBenchmark(Benchmark):
    name = "math_reasoning"
    description = "GSM8K-style word problems; score by extracting the final number."

    def __init__(self, n_samples: int | None = None, **kwargs: object) -> None:
        super().__init__(n_samples, **kwargs)

    @property
    def samples(self) -> list[Sample]:
        out: list[Sample] = []
        for d in load_json_or_jsonl(_DATA):
            out.append(
                Sample(
                    id=d["id"],
                    prompt=_INSTRUCTION + "\n\n" + d["prompt"],
                    expected=d["expected"],
                    grader="regex",
                    request_kwargs=d.get("request_kwargs", {}),
                )
            )
        return out

    async def grade(self, sample, output, *, client, judge_model):  # type: ignore[override]
        return _grade_math(output, str(sample.expected))
