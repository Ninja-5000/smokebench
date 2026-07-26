"""Auto-detect protocol for a given base URL."""

from __future__ import annotations

import httpx

from smoke_bench.clients.anthropic_compat import AnthropicCompatClient
from smoke_bench.clients.base import LLMClient
from smoke_bench.clients.openai_compat import OpenAICompatClient


async def detect_protocol(base_url: str, api_key: str) -> str:
    """Return ``"anthropic"`` or ``"openai"`` based on probe results."""
    # Anthropic exposes /v1/messages with x-api-key. Try that first if a key
    # is present and the URL hints at Anthropic.
    anthropic = AnthropicCompatClient(base_url, api_key)
    openai = OpenAICompatClient(base_url, api_key)
    try:
        if await anthropic.probe():
            return "anthropic"
    except httpx.HTTPError:
        pass
    try:
        if await openai.probe():
            return "openai"
    except httpx.HTTPError:
        pass
    # Default to OpenAI which is the most common.
    return "openai"


def make_client(protocol: str, base_url: str, api_key: str) -> LLMClient:
    p = (protocol or "openai").lower()
    if p == "anthropic":
        return AnthropicCompatClient(base_url, api_key)
    return OpenAICompatClient(base_url, api_key)
