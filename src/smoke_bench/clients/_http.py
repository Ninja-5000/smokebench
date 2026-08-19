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
        "User-Agent": "smokebench/1.0",
    }
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=10.0),
    )


async def abort_client(client: httpx.AsyncClient) -> None:
    """Force-close an httpx client, killing any in-flight TCP socket.

    Plain ``client.aclose()`` only tears down connections tracked by the pool;
    a request cancelled mid-await (e.g. after a timeout) can leave the socket
    orphaned so the server never sees the disconnect and keeps generating.
    Closing the transport directly guarantees the socket is terminated.
    """
    try:
        await client._transport.aclose()
    except Exception:  # noqa: BLE001 - best-effort abort, never mask the caller
        pass
