"""Benchmark selection screen."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Static

from llm_bench.benchmarks import ALL_BENCHMARKS
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


class BenchmarksScreen(Screen):
    """Pick which benchmarks to run."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("4 / 6 — Benchmarks", classes="section-title"),
            Static(
                "Toggle the benchmarks you want to run, then add a custom one or "
                "tune sample counts in Advanced.",
                classes="help",
            ),
            Vertical(id="boxes"),
            Horizontal(
                Button("Add custom benchmark", id="add_custom", variant="default"),
                Button("Advanced (sample counts)", id="advanced", variant="default"),
            ),
            Static("", id="status"),
            Horizontal(
                Button("← Back", id="back", variant="default"),
                Button("Next →", id="next", variant="success"),
            ),
            HelpBar("Ctrl+N next · Esc back"),
            id="screen",
        )

    def on_mount(self) -> None:
        boxes = self.query_one("#boxes", Vertical)
        state: AppState = self.app.state  # type: ignore[attr-defined]
        # Pre-select all on first visit, then respect user's choices
        if not state.selected_benchmarks:
            state.selected_benchmarks = [cls().name for cls in ALL_BENCHMARKS]
        for cls in ALL_BENCHMARKS:
            inst = cls()
            default = inst.name in state.selected_benchmarks
            boxes.mount(
                Checkbox(
                    f"{inst.name} — {inst.description}",
                    value=default,
                    id=f"bench_{inst.name}",
                )
            )
        self._refresh_status("Pick one or more benchmarks. Defaults are pre-selected.")

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "add_custom":
            from llm_bench.tui.screens.custom_bench import CustomBenchScreen

            self.app.push_screen(CustomBenchScreen())
        elif event.button.id == "advanced":
            from llm_bench.tui.screens.advanced import AdvancedScreen

            self.app.push_screen(AdvancedScreen())
        elif event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "next":
            self._advance()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_next_screen(self) -> None:
        self._advance()

    def _advance(self) -> None:
        state: AppState = self.app.state
        chosen: list[str] = []
        boxes = self.query_one("#boxes", Vertical)
        for cb in boxes.children:
            if not isinstance(cb, Checkbox):
                continue
            if not cb.id or not cb.id.startswith("bench_"):
                continue
            name = cb.id[6:]  # strip "bench_"
            if cb.value:
                chosen.append(name)
        for b in state.custom_benchmarks:
            chosen.append(b.name)
        if not chosen:
            self._refresh_status("Select at least one benchmark.", "error")
            return
        state.selected_benchmarks = chosen
        # Check if any selected benchmark requires a judge but none is configured
        judge_only_benchmarks = {"code_explanation", "creative_writing"}
        selected_judge_benchmarks = [b for b in chosen if b in judge_only_benchmarks]
        state_judge_configured = state.judge_model is not None or state.use_separate_judge
        if selected_judge_benchmarks and not state_judge_configured:
            self._refresh_status(
                f"These benchmarks need an LLM judge: {', '.join(selected_judge_benchmarks)}. Go back to enable one.",
                "error",
            )
            return
        from llm_bench.tui.screens.advanced import AdvancedScreen

        self.app.push_screen(AdvancedScreen())
