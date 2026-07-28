"""Benchmark framework: defines tasks, samples, and scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smoke_bench.clients.base import ChatRequest, LLMClient
from smoke_bench.judging.deterministic import GradeResult


@dataclass
class Sample:
    """A single benchmark sample."""

    id: str
    prompt: str
    system: str | None = None
    expected: Any = None
    grader: str = "exact"  # "exact"|"regex"|"contains"|"json_schema"|"rouge_l"|"numeric"|"cosine"|"judge"|"sandbox"
    # For sandbox: code that runs after the model's code
    test_code: str | None = None
    # For judge: rubric
    rubric: str | None = None
    # For judge: reference answer
    reference: str | None = None
    # For json_schema: schema dict
    schema: dict[str, Any] | None = None
    # Extra kwargs for the request (max_tokens, temperature, etc.)
    request_kwargs: dict[str, Any] = field(default_factory=dict)
    # Tags (e.g. "context_size" for needle-in-haystack)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    sample_id: str
    output: str
    score: float
    passed: bool
    latency_s: float
    ttft_s: float | None
    input_tokens: int
    output_tokens: int
    detail: str = ""
    error: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskScore:
    task: str
    n: int
    passed: int
    mean_score: float
    median_latency_s: float
    p95_latency_s: float
    mean_tokens_per_s: float
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float
    per_sample: list[SampleResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


ProgressCallback = Callable[[str, int, int], Awaitable[None]]
"""``await cb(task, completed, total)`` after each sample."""


class Benchmark(ABC):
    """Abstract benchmark task."""

    name: str = "abstract"
    description: str = ""

    def __init__(
        self,
        n_samples: int | None = None,
        max_tokens_override: int | None = None,
        global_max_tokens: int | None = None,
    ) -> None:
        self.n_samples_override = n_samples
        self.max_tokens_override = max_tokens_override
        self.global_max_tokens = global_max_tokens or 4096

    @property
    @abstractmethod
    def samples(self) -> Sequence[Sample]: ...

    def sliced_samples(self) -> list[Sample]:
        all_samples = list(self.samples)
        if self.n_samples_override is not None:
            return all_samples[: self.n_samples_override]
        return all_samples

    async def grade(
        self, sample: Sample, output: str, *, client: LLMClient, judge_model: str | None
    ) -> GradeResult:
        """Dispatch to the right grader."""
        from smoke_bench.judging.deterministic import (
            grade_contains,
            grade_cosine_sim,
            grade_exact,
            grade_json_schema,
            grade_numeric,
            grade_regex,
            grade_rouge_l,
        )
        from smoke_bench.judging.llm_judge import JudgeSpec, judge_output
        from smoke_bench.judging.sandbox import run_python

        g = sample.grader
        if g == "exact":
            return grade_exact(output, str(sample.expected))
        if g == "regex":
            return grade_regex(output, str(sample.expected))
        if g == "contains":
            return grade_contains(output, str(sample.expected))
        if g == "json_schema":
            return grade_json_schema(output, sample.schema or {})
        if g == "rouge_l":
            return grade_rouge_l(output, str(sample.expected))
        if g == "numeric":
            try:
                expected = float(sample.expected)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return GradeResult(score=0.0, passed=False, detail="bad expected")
            return grade_numeric(output, expected)
        if g == "cosine":
            return grade_cosine_sim(output, str(sample.expected))
        if g == "judge":
            if not judge_model:
                return GradeResult(
                    score=0.0, passed=False, detail="no judge model configured"
                )
            spec = JudgeSpec(rubric=sample.rubric or "", reference=sample.reference)
            return await judge_output(
                client=client,
                judge_model=judge_model,
                prompt=sample.prompt,
                output=output,
                spec=spec,
            )
        if g == "sandbox":
            if not sample.test_code:
                return GradeResult(score=0.0, passed=False, detail="no test_code")
            res = run_python(output, sample.test_code)
            return GradeResult(
                score=1.0 if res.passed else 0.0,
                passed=res.passed,
                detail=res.detail + " | " + res.stderr[-200:].strip(),
            )
        return GradeResult(score=0.0, passed=False, detail=f"unknown grader: {g}")

    def build_request(self, sample: Sample, model: str) -> ChatRequest:
        from smoke_bench.clients.base import ChatMessage

        kwargs = {"model": model, "messages": [ChatMessage(role="user", content=sample.prompt)]}
        if sample.system:
            kwargs["system"] = sample.system
        kwargs.update(sample.request_kwargs)
        # Apply per-benchmark token override (from UI) over the sample's own value.
        if self.max_tokens_override is not None:
            kwargs["max_tokens"] = self.max_tokens_override
        # Sensible defaults.
        kwargs.setdefault("max_tokens", self.global_max_tokens)
        kwargs.setdefault("temperature", 0.0)
        if sample.grader == "json_schema":
            kwargs["json_mode"] = True
        return ChatRequest(**kwargs)
