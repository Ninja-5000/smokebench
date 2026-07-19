"""Discover models exposed by an endpoint, with manual fallback."""

from __future__ import annotations

from dataclasses import dataclass

from llm_bench.clients.base import LLMClient, ModelInfo
from llm_bench.clients.detect import detect_protocol, make_client


@dataclass
class DiscoveryResult:
    models: list[ModelInfo]
    protocol: str
    used_fallback: bool = False


async def fetch_models(
    base_url: str,
    api_key: str,
    protocol: str = "auto",
) -> DiscoveryResult:
    """Fetch models from the endpoint. If the endpoint doesn't expose any,
    return an empty list (caller can then prompt for manual entry)."""
    chosen = protocol
    if protocol == "auto":
        chosen = await detect_protocol(base_url, api_key)
    client: LLMClient = make_client(chosen, base_url, api_key)
    # Probe first to verify connectivity
    if not await client.probe():
        return DiscoveryResult(models=[], protocol=chosen, used_fallback=False)
    try:
        models = await client.list_models()
    except Exception:
        return DiscoveryResult(models=[], protocol=chosen, used_fallback=False)
    return DiscoveryResult(models=models, protocol=chosen, used_fallback=False)


def parse_manual_models(spec: str) -> list[ModelInfo]:
    """Parse a comma/newline separated list of model ids into ``ModelInfo``."""
    ids: list[str] = []
    for line in spec.replace(",", "\n").splitlines():
        token = line.strip()
        if token and token not in ids:
            ids.append(token)
    return [ModelInfo(id=i) for i in ids]
