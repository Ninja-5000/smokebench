"""Tests for benchmark task construction and grading dispatch."""

from __future__ import annotations

import pytest

from llm_bench.benchmarks import (
    ALL_BENCHMARKS,
    CodeGenBenchmark,
    CreativeWritingBenchmark,
    CustomBenchmark,
    LongContextBenchmark,
    MathReasonBenchmark,
    instantiate,
)
from llm_bench.benchmarks.base import Sample
from llm_bench.benchmarks.custom import CustomSpec
from llm_bench.judging.llm_judge import JudgeSpec
from llm_bench.judging.sandbox import run_python


def test_all_benchmarks_have_samples() -> None:
    for cls in ALL_BENCHMARKS:
        inst = cls()
        assert inst.name
        assert inst.description
        assert len(inst.samples) > 0


def test_math_benchmark_default_size() -> None:
    b = MathReasonBenchmark()
    assert len(b.samples) >= 10
    s = b.samples[0]
    assert s.grader == "regex"


def test_code_gen_benchmark_passes_for_correct_code() -> None:
    b = CodeGenBenchmark(n_samples=1)
    s = b.samples[0]
    assert s.grader == "sandbox"
    assert s.test_code is not None
    # Emulate an output that solves the first task.
    if s.id == "he_1":
        out = "```python\ndef add(a, b):\n    return a + b\n```"
        res = run_python(out, s.test_code, timeout_s=5.0)
        assert res.passed, res.stderr


def test_long_context_benchmark_sizes() -> None:
    b = LongContextBenchmark()
    assert len(b.samples) >= 3
    # Sizes should be increasing
    sizes = [s.tags["context_words"] for s in b.samples]
    assert sizes == sorted(sizes)


def test_custom_benchmark_from_dict() -> None:
    spec = CustomSpec.from_dict(
        {
            "name": "smoke",
            "description": "smoke test",
            "samples": [
                {"id": "s1", "prompt": "Say hi", "expected": "hi", "grader": "contains"},
            ],
        }
    )
    b = CustomBenchmark(spec)
    assert b.name == "smoke"
    assert len(b.samples) == 1


def test_instantiate_known() -> None:
    assert instantiate("math_reasoning") is not None
    assert instantiate("nonexistent") is None


@pytest.mark.asyncio
async def test_grade_dispatch_for_sandbox() -> None:
    b = CodeGenBenchmark(n_samples=1)
    s = b.samples[0]
    # Force sample content to be valid Python for the first code-gen task.
    if s.id == "he_1":
        out = "def add(a, b):\n    return a + b\n"
        grade = await b.grade(s, out, client=None, judge_model=None)
        assert grade.passed


@pytest.mark.asyncio
async def test_grade_dispatch_for_judge_without_model() -> None:
    b = CreativeWritingBenchmark(n_samples=1)
    s = b.samples[0]
    grade = await b.grade(s, "anything", client=None, judge_model=None)
    assert not grade.passed
    assert "no judge model" in grade.detail
