"""Shared state passed between TUI screens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_bench.benchmarks.base import Benchmark
from llm_bench.clients.base import ModelInfo


@dataclass
class AppState:
    base_url: str = ""
    api_key: str = ""
    protocol: str = "auto"
    protocol_detected: str = "openai"

    models: list[ModelInfo] = field(default_factory=list)
    selected_models: list[str] = field(default_factory=list)
    used_fallback: bool = False

    selected_benchmarks: list[str] = field(default_factory=list)
    custom_benchmarks: list[Benchmark] = field(default_factory=list)

    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_protocol: str | None = None
    use_separate_judge: bool = False

    max_concurrency: int = 4
    sample_overrides: dict[str, int] = field(default_factory=dict)

    # Pricing (loaded later)
    pricing: Any = None

    # Final results
    run_result: Any = None
