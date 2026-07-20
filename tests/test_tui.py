"""Tests for the TUI screens using Textual's pilot harness."""

from __future__ import annotations

import re

import pytest
from textual.widgets import Button, Input

from llm_bench.app import LLMBenchApp
from llm_bench.tui.screens.run import _sanitize_id
from llm_bench.tui.state import AppState


@pytest.mark.asyncio
async def test_app_starts_on_endpoint_screen() -> None:
    app = LLMBenchApp(AppState())
    async with app.run_test() as pilot:
        # Endpoint screen has the "Base URL" label.
        from llm_bench.tui.screens.endpoint import EndpointScreen

        await pilot.pause()
        assert isinstance(app.screen, EndpointScreen)


@pytest.mark.asyncio
async def test_app_quits_cleanly() -> None:
    app = LLMBenchApp(AppState())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        # If we get here without hanging, exit is clean.


@pytest.mark.parametrize(
    "raw",
    [
        "microsoft/phi-4-reasoning-plus",
        "qwen/qwen3.5-9b",
        "nemotron-3-super:cloud",
        "text-embedding-nomic-embed-text-v1.5",
        "essentialai/rnj-1",
        "123-model",
        "model with spaces",
        "model+plus",
    ],
)
def test_sanitize_id_is_valid_identifier(raw: str) -> None:
    sanitized = _sanitize_id(raw)
    assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", sanitized), (
        f"_sanitize_id({raw!r}) -> {sanitized!r} is not a valid Textual id"
    )


@pytest.mark.asyncio
async def test_advanced_screen_saves_sample_overrides() -> None:
    """Regression: ensure the per-benchmark input IDs match between mount and read."""
    from llm_bench.benchmarks import ALL_BENCHMARKS
    from llm_bench.tui.screens.advanced import AdvancedScreen

    state = AppState()
    state.selected_benchmarks = [b().name for b in ALL_BENCHMARKS]
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(AdvancedScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, AdvancedScreen)
        # Set custom override on the first benchmark's input and click save.
        first = ALL_BENCHMARKS[0]()
        screen.query_one(f"#n_{first.name}", Input).value = "7"
        screen.query_one("#concurrency", Input).value = "2"
        screen.query_one("#save", Button).press()
        await pilot.pause()
        assert state.sample_overrides[first.name] == 7
        assert state.max_concurrency == 2


@pytest.mark.asyncio
async def test_endpoint_connection_warns_on_empty_models(monkeypatch) -> None:
    """A bad URL or a non-/models endpoint should not show green 'Connected'."""
    from llm_bench.discovery import DiscoveryResult
    from llm_bench.tui.screens.endpoint import EndpointScreen

    async def fake_fetch(base_url, api_key, protocol):
        return DiscoveryResult(models=[], protocol="openai", used_fallback=False, probe_ok=True)

    state = AppState()
    state.base_url = "http://example.com"
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EndpointScreen)
        monkeypatch.setattr(
            "llm_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )
        await screen.action_test_connection()
        await pilot.pause()
        from textual.widgets import Static

        w = screen.query_one("#status", Static)
        assert "warning" in w.classes, f"expected warning class, got {w.classes!r}"
        # Should NOT say "Connected" when no models
        assert "Connected" not in str(w)


@pytest.mark.asyncio
async def test_endpoint_connection_succeeds_with_models(monkeypatch) -> None:
    from llm_bench.clients.base import ModelInfo
    from llm_bench.discovery import DiscoveryResult
    from llm_bench.tui.screens.endpoint import EndpointScreen

    async def fake_fetch(base_url, api_key, protocol):
        return DiscoveryResult(
            models=[ModelInfo(id="gpt-4o"), ModelInfo(id="gpt-4o-mini")],
            protocol="openai",
            used_fallback=False,
            probe_ok=True,
        )

    state = AppState()
    state.base_url = "http://example.com"
    state.api_key = "sk-test"
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EndpointScreen)
        monkeypatch.setattr(
            "llm_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )
        await screen.action_test_connection()
        await pilot.pause()
        from textual.widgets import Static

        w = screen.query_one("#status", Static)
        assert "success" in w.classes, f"expected success class, got {w.classes!r}"


@pytest.mark.asyncio
async def test_endpoint_connection_shows_error_on_probe_failure(monkeypatch) -> None:
    """When probe fails (bad URL), show 'error' not 'warning'."""
    from llm_bench.discovery import DiscoveryResult
    from llm_bench.tui.screens.endpoint import EndpointScreen

    async def fake_fetch(base_url, api_key, protocol):
        return DiscoveryResult(models=[], protocol="openai", used_fallback=False, probe_ok=False)

    state = AppState()
    state.base_url = "http://invalid"
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EndpointScreen)
        monkeypatch.setattr(
            "llm_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )
        await screen.action_test_connection()
        await pilot.pause()
        from textual.widgets import Static

        w = screen.query_one("#status", Static)
        assert "error" in w.classes, f"expected error class, got {w.classes!r}"
