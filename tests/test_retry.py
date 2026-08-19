"""Tests for the exponential-backoff retry helper."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from smoke_bench.retry import RetryConfig, retry_request


@pytest.mark.asyncio
async def test_retries_transient_500_then_succeeds() -> None:
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://x/v1/chat/completions"),
                response=httpx.Response(500),
            )
        return "ok"

    assert (
        await retry_request(flaky, config=RetryConfig(max_retries=3, base_delay=0.0))
        == "ok"
    )
    assert calls == 3


@pytest.mark.asyncio
async def test_retries_exhausted_raises() -> None:
    calls = 0

    async def always_fails():
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(
            "connection refused", request=httpx.Request("POST", "http://x")
        )

    with pytest.raises(httpx.ConnectError):
        await retry_request(
            always_fails, config=RetryConfig(max_retries=3, base_delay=0.0)
        )
    assert calls == 4


@pytest.mark.asyncio
async def test_non_retryable_400_raises_immediately() -> None:
    calls = 0

    async def bad_request():
        nonlocal calls
        calls += 1
        raise httpx.HTTPStatusError(
            "400 Bad Request",
            request=httpx.Request("POST", "http://x"),
            response=httpx.Response(400),
        )

    with pytest.raises(httpx.HTTPStatusError):
        await retry_request(
            bad_request, config=RetryConfig(max_retries=3, base_delay=0.0)
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_on_retry_callback_receives_attempt_and_error() -> None:
    attempts: list[int] = []
    errors: list[str] = []

    async def flaky():
        raise httpx.ReadTimeout("timed out")

    async def on_retry(attempt: int, exc: Exception) -> None:
        attempts.append(attempt)
        errors.append(type(exc).__name__)

    with pytest.raises(httpx.ReadTimeout):
        await retry_request(
            flaky, config=RetryConfig(max_retries=2, base_delay=0.0), on_retry=on_retry
        )
    assert attempts == [1, 2]
    assert errors == ["ReadTimeout", "ReadTimeout"]


@pytest.mark.asyncio
async def test_retries_respect_backoff_delay_schedule() -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        raise httpx.HTTPStatusError(
            "503",
            request=httpx.Request("POST", "http://x"),
            response=httpx.Response(503),
        )

    original_sleep = asyncio.sleep
    asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await retry_request(
                flaky, config=RetryConfig(max_retries=3, base_delay=2.0)
            )
    finally:
        asyncio.sleep = original_sleep  # type: ignore[assignment]
    assert sleeps == [2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_per_attempt_timeout_cancels_request_before_retry() -> None:
    """A request that exceeds the per-attempt deadline is cancelled (aborting
    any in-flight HTTP work) before the next retry starts, not left running."""
    attempts = 0
    cancelled_attempts: list[int] = []

    async def hang_then_ok():
        nonlocal attempts
        attempts += 1
        try:
            if attempts == 1:
                await asyncio.sleep(3600)  # never completes on its own
            return "ok"
        except asyncio.CancelledError:
            cancelled_attempts.append(attempts)
            raise

    result = await retry_request(
        hang_then_ok,
        config=RetryConfig(max_retries=3, base_delay=0.0, per_attempt_timeout=0.05),
    )
    assert result == "ok"
    assert attempts == 2
    assert cancelled_attempts == [1]  # first attempt was cancelled before retrying


@pytest.mark.asyncio
async def test_per_attempt_timeout_exhausted_raises() -> None:
    """When every attempt exceeds the per-attempt deadline, the run raises a
    retryable httpx timeout after cancelling the last in-flight attempt."""
    cancelled_attempts: list[int] = []

    async def always_hangs():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled_attempts.append(True)
            raise

    with pytest.raises(httpx.ReadTimeout):
        await retry_request(
            always_hangs,
            config=RetryConfig(max_retries=2, base_delay=0.0, per_attempt_timeout=0.05),
        )
    assert cancelled_attempts == [True, True, True]  # every attempt cancelled


@pytest.mark.asyncio
async def test_per_attempt_timeout_on_timeout_fires_before_cancel() -> None:
    """``on_timeout`` must run while the attempt is still running (uncancelled),
    so the caller can abort the in-flight socket before cancellation would make
    that abort ineffective."""
    attempt_cancelled = False
    saw_cancelled_during_timeout: bool | None = None

    async def hang():
        nonlocal attempt_cancelled
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            attempt_cancelled = True
            raise

    async def on_timeout() -> None:
        nonlocal saw_cancelled_during_timeout
        saw_cancelled_during_timeout = attempt_cancelled

    with pytest.raises(httpx.ReadTimeout):
        await retry_request(
            hang,
            config=RetryConfig(max_retries=0, base_delay=0.0, per_attempt_timeout=0.05),
            on_timeout=on_timeout,
        )
    assert saw_cancelled_during_timeout is False  # still uncancelled when hook ran
    assert attempt_cancelled is True  # cancelled right after the hook
