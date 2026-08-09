"""Async orchestrator: runs (model, benchmark) pairs with progress events."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from smoke_bench.benchmarks.base import Benchmark, SampleResult, TaskScore
from smoke_bench.clients.base import LLMClient
from smoke_bench.config import PricingConfig
from smoke_bench.pricing import cost_for

EventKind = str  # "model_start" | "task_start" | "sample_done" | "task_done" | "model_done" | "error"
EventCallback = Callable[[EventKind, dict], Awaitable[None]]


def _estimate_tokens(text: str) -> int:
    """Estimate output tokens when the API omits usage data. tiktoken first."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 - tiktoken may be missing or no model
        return len(text.split())


@dataclass
class RunResult:
    """Aggregated results for a single run."""

    by_model_task: dict[tuple[str, str], TaskScore] = field(default_factory=dict)
    by_model: dict[str, list[TaskScore]] = field(default_factory=lambda: defaultdict(list))
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (model, task, msg)


async def run_benchmark(
    benchmark: Benchmark,
    model: str,
    bench_client: LLMClient,
    judge_client: LLMClient | None = None,
    judge_model: str | None = None,
    on_event: EventCallback | None = None,
    *,
    max_concurrency: int = 4,
    pause_event: asyncio.Event | None = None,
) -> TaskScore:
    """Run a single benchmark for a single model with bounded concurrency."""

    samples = benchmark.sliced_samples()
    sem = asyncio.Semaphore(max_concurrency)
    results: list[SampleResult] = []
    completed = 0
    total = len(samples)

    async def _run_one(sample):
        nonlocal completed
        if pause_event is not None:
            await pause_event.wait()
        request = benchmark.build_request(sample, model)
        start = time.perf_counter()
        ttft: float | None = None
        output_text = ""
        in_tok = out_tok = 0
        err: str | None = None
        client = bench_client
        grading_client = judge_client or bench_client
        resp_latency: float = 0.0
        try:
            if benchmark.name == "latency":
                async with sem:
                    async for chunk in client.stream(request):
                        if chunk.ttft and ttft is None:
                            ttft = time.perf_counter() - start
                        if chunk.delta:
                            output_text += chunk.delta
                        if chunk.usage:
                            in_tok = chunk.usage.input_tokens or in_tok
                            out_tok = chunk.usage.output_tokens or out_tok
            else:
                async with sem:
                    resp = await client.chat(request)
                    output_text = resp.text
                    in_tok = resp.usage.input_tokens
                    out_tok = resp.usage.output_tokens
                    resp_latency = resp.latency_s
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        if not out_tok and output_text:
            out_tok = _estimate_tokens(output_text)
        latency = resp_latency if resp_latency > 0 else time.perf_counter() - start
        gen_latency = latency - (ttft or 0.0)
        grade = await benchmark.grade(
            sample, output_text, client=grading_client, judge_model=judge_model
        )
        completed += 1
        res = SampleResult(
            sample_id=sample.id,
            output=output_text,
            score=grade.score,
            passed=grade.passed,
            latency_s=latency,
            ttft_s=ttft,
            input_tokens=in_tok,
            output_tokens=out_tok,
            detail=grade.detail,
            error=err,
            tags=sample.tags,
        )
        if on_event is not None:
            await on_event(
                "sample_done",
                {
                    "task": benchmark.name,
                    "model": model,
                    "completed": completed,
                    "total": total,
                    "sample_id": sample.id,
                    "passed": res.passed,
                    "score": res.score,
                    "latency_s": res.latency_s,
                    "detail": res.detail,
                    "tokens_per_s": (
                        res.output_tokens / gen_latency
                        if gen_latency > 0 and res.output_tokens
                        else 0.0
                    ),
                },
            )
        return res

    tasks = [asyncio.create_task(_run_one(s)) for s in samples]
    results = await asyncio.gather(*tasks)
    if on_event is not None:
        await on_event(
            "task_done",
            {"task": benchmark.name, "model": model, "n": len(results)},
        )

    # Aggregate
    latencies = sorted(r.latency_s for r in results if r.latency_s > 0)
    if latencies:
        median = latencies[len(latencies) // 2]
        p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    else:
        median = p95 = 0.0
    tps_list = [
        r.output_tokens / max(r.latency_s - (r.ttft_s or 0.0), 0.001)
        for r in results
        if r.latency_s > 0 and r.output_tokens
    ]
    mean_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    return TaskScore(
        task=benchmark.name,
        n=len(results),
        passed=sum(1 for r in results if r.passed),
        mean_score=sum(r.score for r in results) / len(results) if results else 0.0,
        median_latency_s=median,
        p95_latency_s=p95,
        mean_tokens_per_s=mean_tps,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        cost_usd=0.0,  # filled in below by orchestrator
        per_sample=list(results),
    )


async def run_all(
    models: list[str],
    benchmarks: list[Benchmark],
    make_client: Callable[[str], LLMClient],
    pricing: PricingConfig,
    judge_model: str | None = None,
    make_judge_client: Callable[[str], LLMClient] | None = None,
    on_event: EventCallback | None = None,
    *,
    max_concurrency: int = 4,
    pause_event: asyncio.Event | None = None,
) -> RunResult:
    """Run every (model, benchmark) pair. Clients are created lazily per-model."""
    result = RunResult()

    for model in models:
        client = make_client(model)
        judge_client = make_judge_client(model) if make_judge_client else client
        try:
            if on_event is not None:
                await on_event("model_start", {"model": model})
            for bench in benchmarks:
                try:
                    if on_event is not None:
                        await on_event("task_start", {"task": bench.name, "model": model})
                    score = await run_benchmark(
                        bench,
                        model,
                        client,
                        judge_client=judge_client,
                        judge_model=judge_model,
                        on_event=on_event,
                        max_concurrency=max_concurrency,
                        pause_event=pause_event,
                    )
                    # Backfill cost
                    score.cost_usd = cost_for(model, _usage_from_score(score), pricing)
                    result.by_model_task[(model, bench.name)] = score
                    result.by_model[model].append(score)
                except Exception as e:  # noqa: BLE001
                    result.errors.append((model, bench.name, f"{type(e).__name__}: {e}"))
                    if on_event is not None:
                        await on_event("error", {"model": model, "task": bench.name, "msg": str(e)})
            if on_event is not None:
                await on_event("model_done", {"model": model})
        finally:
            await _close_client(client)

    return result


def _usage_from_score(score: TaskScore):  # type: ignore[no-untyped-def]
    from smoke_bench.clients.base import TokenUsage

    return TokenUsage(
        input_tokens=score.total_input_tokens,
        output_tokens=score.total_output_tokens,
    )


async def _close_client(client: LLMClient) -> None:
    """No-op for now; concrete clients don't hold long-lived resources."""
    return None
