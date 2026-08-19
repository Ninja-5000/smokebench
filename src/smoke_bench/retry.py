"""Exponential-backoff retry helper for transient HTTP failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar, final

import httpx

T = TypeVar("T")

# Errors worth retrying: HTTP 5xx (server still loading, proxy 503, overload),
# 429 rate limits, timeouts, and connection failures. 4xx client errors are
# permanent and never retried.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@final
@dataclass
class RetryConfig:
    """Controls how often and how long to retry transient failures."""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_status_codes: frozenset[int] = field(default_factory=lambda: RETRYABLE_STATUS_CODES)
    per_attempt_timeout: float | None = None
    """Hard deadline per attempt; on expiry the in-flight request is aborted
    (via the caller's ``on_timeout`` hook, if any) and then cancelled before the
    next retry starts, so the backend stops generating the abandoned request
    and retries never overlap it."""


def _is_retryable(exc: BaseException, config: RetryConfig) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in config.retryable_status_codes
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return True
    return False


async def retry_request(
    func: Callable[..., Awaitable[T]],
    *args,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], Awaitable[None]] | None = None,
    on_timeout: Callable[[], Awaitable[None]] | None = None,
    **kwargs,
) -> T:
    """Run ``func(*args, **kwargs)``, retrying transient failures.

    Non-retryable exceptions (4xx, assertion logs from the model itself) are
    re-raised immediately. ``on_retry`` is awaited right before each sleep and
    receives ``(attempt_number, exception)`` with ``attempt_number`` starting
    at 1.

    When ``config.per_attempt_timeout`` is set, each attempt is bounded by that
    deadline. On expiry ``on_timeout`` (if given) is awaited *first*, while the
    attempt is still running but before it is cancelled. This lets the caller
    abort the in-flight socket while the task is not yet cancelled, so the
    server actually observes the disconnect. Only then is the task cancelled.
    This prevents retries from overlapping the attempt they replace.
    """
    cfg = config or RetryConfig()
    attempts = cfg.max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            coro = func(*args, **kwargs)
            if cfg.per_attempt_timeout is None:
                return await coro
            task = asyncio.create_task(coro)
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), timeout=cfg.per_attempt_timeout
                )
            except TimeoutError as te:
                if on_timeout is not None:
                    try:
                        await on_timeout()
                    except Exception:  # noqa: BLE001 - abort is best-effort
                        pass
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - superseded by the timeout below
                    pass
                raise httpx.ReadTimeout(
                    f"request timed out after {cfg.per_attempt_timeout:g}s"
                ) from te
        except Exception as exc:  # noqa: BLE001 -- CancelledError is not a subclass
            if not _is_retryable(exc, cfg):
                raise
            if attempt == attempts:
                raise
            if on_retry is not None:
                await on_retry(attempt, exc)
            delay = min(cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)), cfg.max_delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover
