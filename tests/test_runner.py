"""Tests for the async orchestrator."""

from __future__ import annotations

import httpx
import pytest
import respx

from smoke_bench.benchmarks import (
    CodeGenBenchmark,
    LatencyBenchmark,
    MathReasonBenchmark,
)
from smoke_bench.clients import OpenAICompatClient
from smoke_bench.runner import run_all


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
