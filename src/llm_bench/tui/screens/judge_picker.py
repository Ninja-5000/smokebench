"""Pick a judge model. Either reuse one of the selected models, configure a
separate endpoint, or skip LLM-as-judge entirely."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


class JudgePickerScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        yield Container(
            Static("4 / 6 — Judge", classes="section-title"),
            Static(
                "Some benchmarks (creative writing, code explanation) use an LLM judge. "
                "Choose a model to act as judge, or skip.",
                classes="help",
            ),
            Vertical(
                Label("Judge source:"),
                RadioSet(
                    RadioButton(
                        "Use one of the selected models (same endpoint)",
                        id="use_selected",
                        value=not state.use_separate_judge and state.judge_model is None,
                    ),
                    RadioButton(
                        "Use a separate endpoint",
                        id="use_separate",
                        value=state.use_separate_judge,
                    ),
                    RadioButton(
                        "Skip LLM-as-judge (deterministic only)",
                        id="skip",
                        value=state.judge_model is None and state.use_separate_judge,
                    ),
                    id="radio",
                ),
                Label("Selected model (when 'use one of the selected models'):"),
                Input(
                    value=state.judge_model or (state.selected_models[0] if state.selected_models else ""),
                    id="judge_model",
                    placeholder="model id",
                ),
                Label("Separate endpoint URL:"),
                Input(
                    value=state.judge_base_url or "",
                    id="judge_url",
                    placeholder="https://api.openai.com/v1",
                ),
                Label("Separate endpoint API key:"),
                Input(
                    value=state.judge_api_key or "",
                    id="judge_key",
                    password=True,
                    placeholder="sk-…",
                ),
            ),
            Static("", id="status"),
            Horizontal(
                Button("← Back", id="back", variant="default"),
                Button("Next →", id="next", variant="success"),
            ),
            HelpBar("Ctrl+N next · Esc back"),
            id="screen",
        )

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "next":
            self._advance()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_next_screen(self) -> None:
        self._advance()

    def _advance(self) -> None:
        state: AppState = self.app.state
        rs = self.query_one("#radio", RadioSet)
        pressed = rs.pressed_button
        choice = pressed.id if pressed else "use_selected"
        if choice == "skip":
            state.judge_model = None
            state.use_separate_judge = False
        elif choice == "use_separate":
            state.use_separate_judge = True
            state.judge_base_url = self.query_one("#judge_url", Input).value.strip() or state.base_url
            state.judge_api_key = self.query_one("#judge_key", Input).value.strip() or state.api_key
            state.judge_model = self.query_one("#judge_model", Input).value.strip() or None
            if not state.judge_model:
                self._refresh_status("Separate judge requires a model id.", "error")
                return
        else:
            state.use_separate_judge = False
            state.judge_model = self.query_one("#judge_model", Input).value.strip() or None
            if not state.judge_model:
                self._refresh_status("Pick a judge model from the selected list.", "error")
                return
        from llm_bench.tui.screens.advanced import AdvancedScreen

        self.app.push_screen(AdvancedScreen())
