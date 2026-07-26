"""Benchmark selection screen."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Static

from smoke_bench.benchmarks import ALL_BENCHMARKS
from smoke_bench.benchmarks.base import Benchmark
from smoke_bench.tui.state import AppState
from smoke_bench.tui.widgets import HelpBar


def _benchmark_needs_judge(name: str, custom_benchmarks: list[Benchmark]) -> bool:
    """Check if a named benchmark uses LLM-as-judge for any sample."""
    # Check built-in benchmarks
    for cls in ALL_BENCHMARKS:
        if cls().name == name:
            return any(s.grader == "judge" for s in cls().samples)
    # Check custom benchmarks
    for cb in custom_benchmarks:
        if cb.name == name:
            return any(s.grader == "judge" for s in cb.samples)
    return False


class BenchmarksScreen(Screen):
    """Pick which benchmarks to run."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("3 / 6 — Benchmarks", classes="section-title"),
            Static(
                "Toggle the benchmarks you want to run, then add a custom one or "
                "tune sample counts in Advanced.",
                classes="help",
            ),
            VerticalScroll(id="boxes"),
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
        boxes = self.query_one("#boxes", VerticalScroll)
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
            from smoke_bench.tui.screens.custom_bench import CustomBenchScreen

            self.app.push_screen(CustomBenchScreen())
        elif event.button.id == "advanced":
            from smoke_bench.tui.screens.advanced import AdvancedScreen

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
        boxes = self.query_one("#boxes", VerticalScroll)
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
        # Route to JudgePicker if any selected benchmark needs LLM judge
        needs_judge = any(
            _benchmark_needs_judge(name, state.custom_benchmarks) for name in chosen
        )
        if needs_judge:
            from smoke_bench.tui.screens.judge_picker import JudgePickerScreen

            self.app.push_screen(JudgePickerScreen())
        else:
            from smoke_bench.tui.screens.run import RunScreen

            self.app.push_screen(RunScreen())
