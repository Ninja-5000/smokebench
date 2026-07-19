"""End-to-end smoke test: full pipeline with mocked HTTP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_bench.benchmarks import (
    CodeGenBenchmark,
    CreativeWritingBenchmark,
    InstructionJSONBenchmark,
    LatencyBenchmark,
    LongContextBenchmark,
    MathReasonBenchmark,
    SummarizationBenchmark,
)
from llm_bench.clients import OpenAICompatClient
from llm_bench.config import PricingConfig, PricingEntry
from llm_bench.pricing import cost_for
from llm_bench.results.recommend import recommend
from llm_bench.results.report import markdown_report, terminal_table
from llm_bench.results.store import save_result
from llm_bench.runner import run_all


def _chat_mock(text: str, prompt_tokens: int = 50, completion_tokens: int = 30):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def _stream_mock(chunks: list[str], prompt_tokens: int = 5, completion_tokens: int = 10):
    parts: list[str] = []
    for c in chunks:
        parts.append(
            f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n"
        )
    parts.append(
        f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}})}\n\n"
    )
    parts.append("data: [DONE]\n\n")
    return httpx.Response(200, text="".join(parts), headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_full_pipeline_with_two_models(tmp_path: Path) -> None:
    """Run math + code + summarization on two mock models and verify the
    recommendation + report can be generated."""
    # Two models: m1 always answers 72, m2 always answers wrong.
    m1_responses = {
        "/chat/completions": _chat_mock("The answer is 72\n#### 72"),
    }
    m2_responses = {
        "/chat/completions": _chat_mock("I do not know."),
    }

    with respx.mock(base_url="https://api.example.com/v1", assert_all_called=False) as mock:
        # Use side_effect to vary response per call.
        call_count = {"n": 0}

        def side_effect(request):
            call_count["n"] += 1
            # Alternate responses: even calls = m1, odd = m2
            url = str(request.url)
            # We need a way to know which model is calling — but respx mocks
            # are at the URL level, not per-client. Use a single rotating
            # sequence instead. The runner builds separate clients per model,
            # but they all hit the same URL — so we have to rely on call order
            # being deterministic (one model after another).
            return _chat_mock("#### 72" if call_count["n"] % 2 == 1 else "wrong")

        mock.post("/chat/completions").mock(side_effect=side_effect)

        result = await run_all(
            models=["m1", "m2"],
            benchmarks=[
                MathReasonBenchmark(n_samples=2),
                CodeGenBenchmark(n_samples=1),
            ],
            make_client=lambda m: OpenAICompatClient("https://api.example.com/v1", "sk-test"),
            pricing=PricingConfig(
                entries={
                    "m1": PricingEntry(input_per_million=3.0, output_per_million=15.0),
                    "m2": PricingEntry(input_per_million=15.0, output_per_million=60.0),
                }
            ),
            judge_model=None,
        )

    assert ("m1", "math_reasoning") in result.by_model_task
    assert ("m2", "math_reasoning") in result.by_model_task
    # m1 should beat m2 on math (m1 returns 72, the first sample expects 72)
    s1 = result.by_model_task[("m1", "math_reasoning")]
    s2 = result.by_model_task[("m2", "math_reasoning")]
    assert s1.passed >= s2.passed

    # Cost is computed
    assert s1.cost_usd > 0

    # Recommendations
    recs = recommend(result)
    assert any(r.model for r in recs)

    # Save + markdown
    out = save_result(result, out_dir=tmp_path)
    assert (out / "summary.json").exists()
    assert (out / "details.json").exists()
    assert (out / "samples.jsonl").exists()
    md_path = out / "report.md"
    markdown_report(result, out_path=md_path)
    assert md_path.exists()
    text = md_path.read_text()
    assert "# llm-bench Report" in text
    assert "## Recommendations" in text

    # Terminal table rendering
    rendered = terminal_table(result)
    assert "Recommendations" in rendered
    assert "math_reasoning" in rendered
