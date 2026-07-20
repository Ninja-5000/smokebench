"""Persistent configuration. Stored at ``~/.llm-bench.yaml`` with 0600 perms."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr

CONFIG_PATH = Path(
    os.environ.get("LLM_BENCH_CONFIG", Path.home() / ".llm-bench.yaml")
)


class EndpointConfig(BaseModel):
    """Configuration for a single endpoint."""

    name: str = "default"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    protocol: str = "auto"  # "auto" | "openai" | "anthropic"
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None
    judge_protocol: str | None = None
    judge_model: str | None = None
    use_separate_judge: bool = False


class PricingEntry(BaseModel):
    """Per-token pricing in USD. ``None`` means unknown."""

    input_per_million: float | None = None
    output_per_million: float | None = None


class PricingConfig(BaseModel):
    """``model_id -> PricingEntry`` mapping."""

    entries: dict[str, PricingEntry] = Field(default_factory=dict)

    def get(self, model_id: str) -> PricingEntry:
        return self.entries.get(model_id, PricingEntry())


class AppConfig(BaseModel):
    """Top-level persisted config."""

    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    last_models: list[str] = Field(default_factory=list)
    last_benchmarks: list[str] = Field(default_factory=list)
    sample_overrides: dict[str, int] = Field(default_factory=dict)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    data: Any = yaml.safe_load(path.read_text()) or {}
    try:
        return AppConfig.model_validate(data)
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json")
    # Unmask SecretStr fields so the YAML roundtrips.
    _unmask(payload, cfg)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _unmask(payload: dict, cfg: AppConfig) -> None:
    """Replace masked SecretStr dumps with the real values (for re-loading)."""
    if "endpoint" in payload and isinstance(payload["endpoint"], dict):
        ep = payload["endpoint"]
        if "api_key" in ep and cfg.endpoint.api_key:
            ep["api_key"] = cfg.endpoint.api_key.get_secret_value()
        if ep.get("judge_api_key") and cfg.endpoint.judge_api_key:
            ep["judge_api_key"] = cfg.endpoint.judge_api_key.get_secret_value()
