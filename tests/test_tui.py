"""Tests for the TUI screens using Textual's pilot harness."""

from __future__ import annotations

import pytest

from llm_bench.app import LLMBenchApp
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
