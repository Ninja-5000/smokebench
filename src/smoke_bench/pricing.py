"""Per-model cost calculation. All costs in USD."""

from __future__ import annotations

from smoke_bench.clients.base import TokenUsage
from smoke_bench.config import PricingConfig


def cost_for(model_id: str, usage: TokenUsage, pricing: PricingConfig | None) -> float:
    """Return the dollar cost for a request, or 0.0 if no pricing entry."""
    if pricing is None:
        return 0.0
    entry = pricing.get(model_id)
    in_rate = entry.input_per_million or 0.0
    out_rate = entry.output_per_million or 0.0
    return (usage.input_tokens / 1_000_000.0) * in_rate + (
        usage.output_tokens / 1_000_000.0
    ) * out_rate
