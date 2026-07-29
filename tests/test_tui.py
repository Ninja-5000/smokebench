"""Tests for the TUI screens using Textual's pilot harness."""

from __future__ import annotations

import re

import pytest
from textual.containers import HorizontalScroll
from textual.widgets import Button, Collapsible, DataTable, Input, ProgressBar, RadioButton, Select, Static

from smoke_bench.app import LLMBenchApp
from smoke_bench.tui.screens.run import _sanitize_id
from smoke_bench.tui.state import AppState


@pytest.mark.asyncio
async def test_app_starts_on_endpoint_screen() -> None:
    app = LLMBenchApp(AppState())
    async with app.run_test() as pilot:
        # Endpoint screen has the "Base URL" label.
        from smoke_bench.tui.screens.endpoint import EndpointScreen

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
    from smoke_bench.benchmarks import ALL_BENCHMARKS
    from smoke_bench.tui.screens.advanced import AdvancedScreen

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
    from smoke_bench.discovery import DiscoveryResult
    from smoke_bench.tui.screens.endpoint import EndpointScreen

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
            "smoke_bench.tui.screens.endpoint.fetch_models", fake_fetch
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
    from smoke_bench.clients.base import ModelInfo
    from smoke_bench.discovery import DiscoveryResult
    from smoke_bench.tui.screens.endpoint import EndpointScreen

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
            "smoke_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )
        await screen.action_test_connection()
        await pilot.pause()
        from textual.widgets import Static

        w = screen.query_one("#status", Static)
        assert "success" in w.classes, f"expected success class, got {w.classes!r}"


@pytest.mark.asyncio
async def test_endpoint_connection_shows_error_on_probe_failure(monkeypatch) -> None:
    """When probe fails (bad URL), show 'error' not 'warning'."""
    from smoke_bench.discovery import DiscoveryResult
    from smoke_bench.tui.screens.endpoint import EndpointScreen

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
            "smoke_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )
        await screen.action_test_connection()
        await pilot.pause()
        from textual.widgets import Static

        w = screen.query_one("#status", Static)
        assert "error" in w.classes, f"expected error class, got {w.classes!r}"


@pytest.mark.asyncio
async def test_endpoint_advance_uses_form_values_after_probe(monkeypatch) -> None:
    from smoke_bench.discovery import DiscoveryResult
    from smoke_bench.tui.screens.endpoint import EndpointScreen
    from smoke_bench.tui.screens.models import ModelsScreen

    async def fake_fetch(base_url, api_key, protocol):
        return DiscoveryResult(models=[{"id": "gpt-4o"}], protocol="openai", used_fallback=False, probe_ok=True)

    state = AppState()
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EndpointScreen)
        monkeypatch.setattr(
            "smoke_bench.tui.screens.endpoint.fetch_models", fake_fetch
        )

        screen.query_one("#base_url", Input).value = "https://example.test/v1"
        screen.query_one("#api_key", Input).value = "sk-test"
        screen.query_one("#protocol", Select).value = "anthropic"

        await screen._advance()
        await pilot.pause()

        assert state.base_url == "https://example.test/v1"
        assert state.api_key == "sk-test"
        assert state.protocol == "anthropic"
        assert isinstance(app.screen, ModelsScreen)


@pytest.mark.asyncio
async def test_run_dashboard_has_overall_model_progress_and_live_result_rows(monkeypatch) -> None:
    """The live dashboard uses one bar per model and appends completed samples."""
    from smoke_bench.tui.screens.run import RunScreen

    async def do_not_run(self):
        return None

    monkeypatch.setattr(RunScreen, "_run", do_not_run)
    state = AppState(
        selected_models=["model-a", "model-b"],
        selected_benchmarks=["math_reasoning"],
    )
    app = LLMBenchApp(state)
    async with app.run_test(size=(140, 36)) as pilot:
        await app.push_screen(RunScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        assert len(list(screen.query(ProgressBar))) == 2
        screen._append_result(
            {
                "sample_id": "gsm8k_1",
                "model": "model-a",
                "passed": True,
                "score": 1.0,
                "latency_s": 1.25,
                "tokens_per_s": 20.0,
            }
        )
        table = screen.query_one("#live-results", DataTable)
        assert table.row_count == 1
        screen._set_model_status("model-a", "Current: math_reasoning")
        assert "Current" in str(screen.query_one("#model_status_model-a", Static).render())


@pytest.mark.asyncio
async def test_run_dashboard_collapses_secondary_panels_on_narrow_terminals(monkeypatch) -> None:
    from smoke_bench.tui.screens.run import RunScreen

    async def do_not_run(self):
        return None

    monkeypatch.setattr(RunScreen, "_run", do_not_run)
    state = AppState(selected_models=["model-a"], selected_benchmarks=["math_reasoning"])
    app = LLMBenchApp(state)
    async with app.run_test(size=(70, 30)) as pilot:
        await app.push_screen(RunScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        assert not screen.query_one("#progress-panel", Collapsible).collapsed
        assert screen.query_one("#results-panel", Collapsible).collapsed
        assert screen.query_one("#log-panel", Collapsible).collapsed
        assert screen.query_one("#model-progress").size.height <= 10
        assert screen.query_one("#run-actions").display


@pytest.mark.asyncio
async def test_run_dashboard_reopens_panels_and_keeps_controls_on_resize(monkeypatch) -> None:
    from smoke_bench.tui.screens.run import RunScreen

    async def do_not_run(self):
        return None

    monkeypatch.setattr(RunScreen, "_run", do_not_run)
    state = AppState(selected_models=["model-a"], selected_benchmarks=["math_reasoning"])
    app = LLMBenchApp(state)
    async with app.run_test(size=(70, 20)) as pilot:
        await app.push_screen(RunScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        assert screen.query_one("#results-panel", Collapsible).collapsed
        assert screen.query_one("#run-actions").display
        await pilot.resize_terminal(140, 36)
        await pilot.pause()
        for panel_id in ("#progress-panel", "#results-panel", "#log-panel"):
            assert not screen.query_one(panel_id, Collapsible).collapsed


@pytest.mark.asyncio
async def test_run_dashboard_medium_layout_has_a_complete_scroll_canvas(monkeypatch) -> None:
    from smoke_bench.tui.screens.run import RunScreen

    async def do_not_run(self):
        return None

    monkeypatch.setattr(RunScreen, "_run", do_not_run)
    state = AppState(selected_models=["model-a"], selected_benchmarks=["math_reasoning"])
    app = LLMBenchApp(state)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.push_screen(RunScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        scroll = screen.query_one("#dashboard-scroll", HorizontalScroll)
        assert scroll.virtual_size.width > scroll.size.width
        for panel_id in ("#progress-panel", "#results-panel", "#log-panel"):
            assert screen.query_one(panel_id, Collapsible).title


@pytest.mark.asyncio
async def test_judge_picker_requires_a_model_for_a_separate_endpoint() -> None:
    from smoke_bench.tui.screens.judge_picker import JudgePickerScreen

    state = AppState(selected_models=["model-a"])
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await app.push_screen(JudgePickerScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, JudgePickerScreen)
        screen.query_one("#use_separate", RadioButton).value = True
        await pilot.pause()
        screen.action_next_screen()
        await pilot.pause()
        assert app.screen is screen
        status = screen.query_one("#status", Static)
        assert "select a model" in str(status.render()).lower()
        assert "error" in status.classes
        assert state.judge_model is None


@pytest.mark.asyncio
async def test_judge_picker_accepts_a_selected_separate_endpoint_model(monkeypatch) -> None:
    from textual.widgets import Select

    from smoke_bench.tui.screens.judge_picker import JudgePickerScreen
    from smoke_bench.tui.screens.run import RunScreen

    async def do_not_run(self):
        return None

    monkeypatch.setattr(RunScreen, "_run", do_not_run)
    state = AppState(selected_models=["model-a"])
    app = LLMBenchApp(state)
    async with app.run_test() as pilot:
        await app.push_screen(JudgePickerScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, JudgePickerScreen)
        screen.query_one("#use_separate", RadioButton).value = True
        judge_select = screen.query_one("#judge_model_sep", Select)
        judge_select.set_options([("separate-judge", "separate-judge")])
        judge_select.value = "separate-judge"
        await pilot.pause()
        screen.action_next_screen()
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        assert state.use_separate_judge
        assert state.judge_model == "separate-judge"
