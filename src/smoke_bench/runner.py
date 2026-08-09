"""Async orchestrator: runs (model, benchmark) pairs with progress events."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from smoke_bench.benchmarks.base import Benchmark, Sample, SampleResult, TaskScore
from smoke_bench.clients.base import LLMClient
from smoke_bench.config import PricingConfig
from smoke_bench.judging.deterministic import GradeResult
from smoke_bench.pricing import cost_for
from smoke_bench.retry import RetryConfig, retry_request

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


def _retry_emitter(
    model: str, task: str, sample_id: str, on_event: EventCallback | None
) -> Callable[[int, Exception], Awaitable[None]] | None:
    """Build an ``on_retry`` callback that surfaces retries via events, if wired."""
    if on_event is None:
        return None

    async def _emit(attempt: int, exc: Exception) -> None:
        await on_event(
            "retry",
            {
                "model": model,
                "task": task,
                "sample_id": sample_id,
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    return _emit


@dataclass
class RunResult:
    """Aggregated results for a single run."""

    by_model_task: dict[tuple[str, str], TaskScore] = field(default_factory=dict)
    by_model: dict[str, list[TaskScore]] = field(default_factory=lambda: defaultdict(list))
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (model, task, msg)


@dataclass
class _Collected:
    """Raw model response captured in the collect phase, before grading."""

    sample: Sample
    output_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None
    error: str | None = None


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
    retry_config: RetryConfig | None = None,
) -> TaskScore:
    """Run a single benchmark for a single model with bounded concurrency.

    The run happens in two phases. First every sample's model request runs
    concurrently (bounded by ``max_concurrency``). Then responses are graded
    sequentially, one after another, so a judge model never receives
    overlapping requests.
    """

    samples = benchmark.sliced_samples()
    sem = asyncio.Semaphore(max_concurrency)
    results: list[SampleResult] = []
    total = len(samples)

    async def _collect_one(sample):
        """Phase 1: request the output from the model under test."""
        request = benchmark.build_request(sample, model)
        client = bench_client
        start = time.perf_counter()
        ttft: float | None = None
        output_text = ""
        in_tok = out_tok = 0
        resp_latency: float = 0.0

        async def _do_request():
            nonlocal ttft, output_text, in_tok, out_tok, resp_latency
            ttft = None
            output_text = ""
            in_tok = out_tok = 0
            resp_latency = 0.0
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

        if pause_event is not None:
            await pause_event.wait()
        try:
            await retry_request(
                _do_request,
                config=retry_config,
                on_retry=_retry_emitter(model, benchmark.name, str(sample.id), on_event),
            )
        except Exception as e:  # noqa: BLE001 - exhausted all retries
            return _Collected(sample=sample, error=f"{type(e).__name__}: {e}")
        if not out_tok and output_text:
            out_tok = _estimate_tokens(output_text)
        latency = resp_latency if resp_latency > 0 else time.perf_counter() - start
        return _Collected(
            sample=sample,
            output_text=output_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            ttft_s=ttft,
        )

    async def _grade_one(col: _Collected) -> SampleResult:
        """Phase 2: grade and report a collected response, one at a time."""
        if pause_event is not None:
            await pause_event.wait()
        sample = col.sample
        grading_client = judge_client or bench_client
        if col.error:
            grade = GradeResult(score=0.0, passed=False, detail=col.error)
        else:
            grade = await retry_request(
                benchmark.grade,
                sample,
                col.output_text,
                client=grading_client,
                judge_model=judge_model,
                config=retry_config,
                on_retry=_retry_emitter(model, benchmark.name, str(sample.id), on_event),
            )
        return SampleResult(
            sample_id=sample.id,
            output=col.output_text,
            score=grade.score,
            passed=grade.passed,
            latency_s=col.latency_s,
            ttft_s=col.ttft_s,
            input_tokens=col.input_tokens,
            output_tokens=col.output_tokens,
            detail=grade.detail,
            error=col.error,
            tags=sample.tags,
        )

    # Phase 1: collect every response concurrently, bounded by the semaphore.
    collected = await asyncio.gather(*(_collect_one(s) for s in samples))
    # Phase 2: grade serially so the judge never gets concurrent requests.
    for col in collected:
        res = await _grade_one(col)
        results.append(res)
        if on_event is not None:
            await on_event(
                "sample_done",
                {
                    "task": benchmark.name,
                    "model": model,
                    "completed": len(results),
                    "total": total,
                    "sample_id": res.sample_id,
                    "passed": res.passed,
                    "score": res.score,
                    "latency_s": res.latency_s,
                    "detail": res.detail,
                    "tokens_per_s": (
                        res.output_tokens / (res.latency_s - (res.ttft_s or 0.0))
                        if (res.latency_s - (res.ttft_s or 0.0)) > 0 and res.output_tokens
                        else 0.0
                    ),
                },
            )
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
    retry_config: RetryConfig | None = None,
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
                        retry_config=retry_config,
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
