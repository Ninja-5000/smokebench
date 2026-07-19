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
