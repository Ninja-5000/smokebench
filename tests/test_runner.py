"""Tests for the async orchestrator."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from smoke_bench.benchmarks import (
    CodeGenBenchmark,
    LatencyBenchmark,
    MathReasonBenchmark,
)
from smoke_bench.benchmarks.base import Benchmark, Sample
from smoke_bench.clients import OpenAICompatClient
from smoke_bench.clients.base import ChatRequest, ChatResponse, LLMClient, ModelInfo, TokenUsage
from smoke_bench.runner import run_all, run_benchmark


def _ok_response(content: str = "#### 42", prompt_tokens: int = 5, completion_tokens: int = 3):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


@pytest.mark.asyncio
async def test_run_math_benchmark() -> None:
    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        mock.post("/chat/completions").mock(return_value=_ok_response("The answer is 72\n#### 72"))
        def client_factory(model):  # noqa: E731
            return OpenAICompatClient("https://api.example.com/v1", "sk-test")
        result = await run_all(
            models=["m1"],
            benchmarks=[MathReasonBenchmark(n_samples=3)],
            make_client=client_factory,
            pricing=None,  # type: ignore[arg-type]
            judge_model=None,
        )
    assert ("m1", "math_reasoning") in result.by_model_task
    score = result.by_model_task[("m1", "math_reasoning")]
    assert score.n == 3
    assert score.passed >= 1


@pytest.mark.asyncio
async def test_run_code_generation_with_correct_code() -> None:
    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        # Provide a code response that solves the first code-gen sample.
        body = (
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        mock.post("/chat/completions").mock(return_value=_ok_response(body))
        result = await run_all(
            models=["m1"],
            benchmarks=[CodeGenBenchmark(n_samples=1)],
            make_client=lambda m: OpenAICompatClient("https://api.example.com/v1", "sk-test"),
            pricing=None,  # type: ignore[arg-type]
        )
    score = result.by_model_task[("m1", "code_generation")]
    assert score.passed == 1


@pytest.mark.asyncio
async def test_run_latency_uses_streaming() -> None:
    body = (
        "data: {\"choices\":[{\"delta\":{\"content\":\"1\\n\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"2\\n\"}}]}\n\n"
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n"
        "data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        result = await run_all(
            models=["m1"],
            benchmarks=[LatencyBenchmark(n_samples=2)],
            make_client=lambda m: OpenAICompatClient("https://api.example.com/v1", "sk-test"),
            pricing=None,  # type: ignore[arg-type]
        )
    score = result.by_model_task[("m1", "latency")]
    assert score.n == 2
    assert score.passed >= 1


@pytest.mark.asyncio
async def test_run_progress_events() -> None:
    events = []

    async def on_event(kind, payload):
        events.append((kind, payload))

    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        mock.post("/chat/completions").mock(return_value=_ok_response("The answer is 72\n#### 72"))
        result = await run_all(
            models=["m1"],
            benchmarks=[MathReasonBenchmark(n_samples=2)],
            make_client=lambda m: OpenAICompatClient("https://api.example.com/v1", "sk-test"),
            pricing=None,  # type: ignore[arg-type]
            on_event=on_event,
        )
    kinds = [k for k, _ in events]
    assert "model_start" in kinds
    assert "task_start" in kinds
    assert "sample_done" in kinds
    assert "task_done" in kinds
    assert "model_done" in kinds
    assert kinds.index("model_start") < kinds.index("task_start") < kinds.index("sample_done")
    assert kinds.count("task_done") == 1
    assert result.errors == []
    score = result.by_model_task[("m1", "math_reasoning")]
    assert score.passed >= 1


@pytest.mark.asyncio
async def test_run_estimates_output_tokens_when_usage_omitted() -> None:
    body = (
        "data: {\"choices\":[{\"delta\":{\"content\":\"seven\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\" eleven\"}}]}\n\n"
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n"
    )
    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
        result = await run_all(
            models=["m1"],
            benchmarks=[LatencyBenchmark(n_samples=1)],
            make_client=lambda m: OpenAICompatClient("https://api.example.com/v1", "sk-test"),
            pricing=None,  # type: ignore[arg-type]
        )
    score = result.by_model_task[("m1", "latency")]
    sample = score.per_sample[0]
    assert sample.output == "seven eleven"
    assert sample.output_tokens > 0
    assert score.mean_tokens_per_s > 0


class _TrackingClient(LLMClient):
    """Records per-call tags and detects overlapping concurrency."""

    def __init__(self, base_url: str, api_key: str):
        super().__init__(base_url, api_key)
        self.calls: list[str] = []
        self._active_judge = 0
        self.max_active_judge = 0

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def probe(self) -> bool:
        return True

    async def chat(self, request: ChatRequest) -> ChatResponse:
        tag = f"judge:{request.model}" if "judge" in request.model else f"model:{request.model}"
        if tag.startswith("judge"):
            self._active_judge += 1
            self.max_active_judge = max(self.max_active_judge, self._active_judge)
            await asyncio.sleep(0)
            self._active_judge -= 1
        self.calls.append(tag)
        if tag.startswith("judge"):
            return ChatResponse(
                text='{"score": 4}',
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                latency_s=0.01,
            )
        return ChatResponse(
            text="some prose response",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            latency_s=0.01,
        )

    def stream(self, request: ChatRequest):  # type: ignore[override]
        raise AssertionError("stream should not be used")


class _JudgeBenchmark(Benchmark):
    """Minimal benchmark whose samples are graded by the judge model."""

    name = "judge_bench"

    def __init__(self, n_samples: int | None = None):
        super().__init__(n_samples)

    @property
    def samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"j{i}",
                prompt="write a short story",
                grader="judge",
                rubric="rate 1-5",
            )
            for i in range(self.n_samples_override or 3)
        ]


@pytest.mark.asyncio
async def test_pipeline_grades_after_all_collected_and_serially() -> None:
    """Model requests all run first; judge calls are sequential (never overlap)."""
    client = _TrackingClient("https://api.example.com/v1", "sk-test")
    bench = _JudgeBenchmark(n_samples=3)
    score = await run_benchmark(bench, "m1", client, judge_client=client, judge_model="judge-x")
    assert score.n == 3

    tags = client.calls
    assert {t.split(":")[0] for t in tags} == {"model", "judge"}
    # All model requests precede every judge request.
    assert tags.index("judge:judge-x") > tags.index("model:m1") + 1
    # Grading is serial: judge calls never overlap.
    assert client.max_active_judge <= 1
    # Each sample is collected once and judged once.
    assert sum(t.startswith("judge:") for t in tags) == 3
    assert sum(t.startswith("model:") for t in tags) == 3


@pytest.mark.asyncio
async def test_model_retries_transient_500_then_succeeds() -> None:
    """A 500 from the model is retried; the run completes with a real result."""
    from smoke_bench.retry import RetryConfig

    calls = 0

    class _FlakyClient(_TrackingClient):
        async def chat(self, request):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.HTTPStatusError(
                    "500",
                    request=httpx.Request("POST", "http://x/v1"),
                    response=httpx.Response(500),
                )
            return await super().chat(request)

    client = _FlakyClient("https://api.example.com/v1", "sk-test")
    bench = MathReasonBenchmark(n_samples=1)
    score = await run_benchmark(
        bench,
        "m1",
        client,
        retry_config=RetryConfig(max_retries=5, base_delay=0.05),
    )
    assert calls == 3
    assert score.n == 1
    assert score.per_sample[0].error is None


@pytest.mark.asyncio
async def test_judge_retry_upholds_score_after_transient_failure() -> None:
    """A transient judge failure is retried until the judge model is ready."""
    from smoke_bench.retry import RetryConfig

    judge_calls = {"n": 0}

    class _FlakyJudge(_TrackingClient):
        async def chat(self, request):
            if "judge" in request.model:
                judge_calls["n"] += 1
                if judge_calls["n"] <= 2:
                    raise httpx.HTTPStatusError(
                        "503",
                        request=httpx.Request("POST", "http://j/v1"),
                        response=httpx.Response(503),
                    )
            return await super().chat(request)

    client = _FlakyJudge("https://api.example.com/v1", "sk-test")
    bench = _JudgeBenchmark(n_samples=1)
    score = await run_benchmark(
        bench,
        "m1",
        client,
        judge_client=client,
        judge_model="judge-x",
        retry_config=RetryConfig(max_retries=5, base_delay=0.05),
    )
    assert judge_calls["n"] == 3
    assert score.n == 1
    assert score.passed == 1


@pytest.mark.asyncio
async def test_model_exhausted_retries_reports_error_and_fails() -> None:
    """When the model never warms up, the sample is recorded as failed."""
    from smoke_bench.retry import RetryConfig

    calls = {"n": 0}

    class _DownClient(_TrackingClient):
        async def chat(self, request):
            calls["n"] += 1
            raise httpx.HTTPStatusError(
                "500",
                request=httpx.Request("POST", "http://x/v1"),
                response=httpx.Response(500),
            )

    client = _DownClient("https://api.example.com/v1", "sk-test")
    bench = MathReasonBenchmark(n_samples=2)
    score = await run_benchmark(
        bench,
        "m1",
        client,
        retry_config=RetryConfig(max_retries=2, base_delay=0.05),
    )
    assert calls["n"] == 6  # 3 attempts x 2 samples
    assert score.n == 2
    assert score.passed == 0
    assert all(r.error is not None for r in score.per_sample)
    assert all(r.score == 0.0 for r in score.per_sample)
