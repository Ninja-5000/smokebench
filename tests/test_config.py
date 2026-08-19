"""Tests for configuration persistence."""

from __future__ import annotations

from pathlib import Path

from smoke_bench.config import AppConfig, EndpointConfig, load_config, save_config


def test_save_and_load(tmp_path: Path) -> None:
    cfg = AppConfig(
        endpoint=EndpointConfig(base_url="https://x", api_key="secret"),
        last_models=["m1"],
        sample_overrides={"math_reasoning": 5},
        per_attempt_timeout=42.0,
    )
    save_config(cfg, tmp_path / "cfg.yaml")
    loaded = load_config(tmp_path / "cfg.yaml")
    assert loaded.endpoint.base_url == "https://x"
    # The loaded config roundtrips the secret value.
    assert loaded.endpoint.api_key.get_secret_value() == "secret"
    assert loaded.last_models == ["m1"]
    assert loaded.sample_overrides["math_reasoning"] == 5
    assert loaded.per_attempt_timeout == 42.0
    # Permissions
    mode = (tmp_path / "cfg.yaml").stat().st_mode & 0o777
    assert mode == 0o600
