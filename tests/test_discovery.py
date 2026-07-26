"""Tests for the discovery helpers."""

from __future__ import annotations

import httpx
import pytest
import respx

from smoke_bench.discovery import fetch_models, parse_manual_models


@pytest.mark.asyncio
async def test_fetch_models_openai() -> None:
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.get("/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]},
            )
        )
        res = await fetch_models("https://api.example.com/v1", "sk-test", protocol="openai")
    assert [m.id for m in res.models] == ["gpt-4o", "gpt-4o-mini"]
    assert res.protocol == "openai"
    assert not res.used_fallback


def test_parse_manual_models() -> None:
    models = parse_manual_models("a, b\nc, a")
    assert [m.id for m in models] == ["a", "b", "c"]
