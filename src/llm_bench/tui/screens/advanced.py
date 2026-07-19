"""Advanced options: per-benchmark sample counts and concurrency."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static

from llm_bench.benchmarks import ALL_BENCHMARKS
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


class AdvancedScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Advanced settings", classes="section-title"),
            Static(
                "Override the default sample count per benchmark, or change the "
                "concurrency level. Leave blank to use defaults.",
                classes="help",
            ),
            Vertical(id="form"),
            Horizontal(
                Button("Save", id="save", variant="success"),
                Button("Cancel", id="cancel", variant="default"),
            ),
            Static("", id="status"),
            HelpBar("Esc back"),
            id="screen",
        )

    def on_mount(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        form = self.query_one("#form", Vertical)
        for cls in ALL_BENCHMARKS:
            inst = cls()
            default = state.sample_overrides.get(inst.name, "")
            form.mount(
                Horizontal(
                    Label(f"{inst.name} (default {len(inst.samples)}):"),
                    Input(value=str(default), id=f"n_{inst.name}", placeholder="default"),
                )
            )
        form.mount(
            Horizontal(
                Label("Concurrency:"),
                Input(value=str(state.max_concurrency), id="concurrency"),
            )
        )

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "save":
            state: AppState = self.app.state  # type: ignore[attr-defined]
            for cls in ALL_BENCHMARKS:
                raw = self.query_one(f"#n_{cls.__name__}", Input).value.strip()
                if raw:
                    try:
                        state.sample_overrides[cls().name] = int(raw)
                    except ValueError:
                        self._refresh_status(
                            f"{cls.__name__}: invalid number '{raw}'", "error"
                        )
                        return
            try:
                state.max_concurrency = max(1, int(self.query_one("#concurrency", Input).value))
            except ValueError:
                state.max_concurrency = 4
            self._refresh_status("Saved.", "success")
            self.app.pop_screen()
        elif event.button.id == "cancel":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()