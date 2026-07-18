"""Helpers for safely using ``httpx`` in async contexts."""

from __future__ import annotations

import httpx


def make_client(base_url: str, api_key: str, timeout: float = 60.0) -> httpx.AsyncClient:
    """Build an authenticated ``httpx.AsyncClient``.

    The returned client does not validate URL prefixes; callers should pass a
    fully-qualified base URL (e.g. ``https://api.openai.com/v1``).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "llm-bench/0.1",
    }
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=10.0),
    )
