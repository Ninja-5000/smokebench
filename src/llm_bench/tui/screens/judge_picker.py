"""Pick a judge model. Either reuse one of the selected models, configure a
separate endpoint, or skip LLM-as-judge entirely."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Select, Static

from llm_bench.discovery import fetch_models
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


class JudgePickerScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+t", "test_connection", "Test connection"),
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        selected = [(m, m) for m in state.selected_models]
        yield Container(
            Static("4 / 6 — Judge", classes="section-title"),
            Static(
                "Some benchmarks use an LLM judge for scoring. "
                "Choose a model to act as judge, or skip.",
                classes="help",
            ),
            Label("Judge source:"),
            RadioSet(
                RadioButton(
                    "Use one of the selected models (same endpoint)",
                    id="use_selected",
                    value=not state.use_separate_judge,
                ),
                RadioButton(
                    "Use a separate endpoint",
                    id="use_separate",
                    value=state.use_separate_judge,
                ),
                RadioButton(
                    "Skip LLM-as-judge (deterministic only)",
                    id="skip",
                    value=False,
                ),
                id="radio",
            ),
            VerticalScroll(
                Label("Select a judge model:"),
                Select(
                    selected,
                    id="judge_model_sel",
                    value=state.judge_model if state.judge_model and state.judge_model in state.selected_models else Select.NULL,
                    allow_blank=False,
                ),
                id="selected_section",
            ),
            VerticalScroll(
                Label("URL:"),
                Input(
                    value=state.judge_base_url or "",
                    id="judge_url",
                    placeholder="https://api.openai.com/v1",
                ),
                Label("API key:"),
                Input(
                    value=state.judge_api_key or "",
                    id="judge_key",
                    password=True,
                    placeholder="sk-…",
                ),
                Button("Test connection", id="test", variant="primary"),
                Label("Select a judge model from the endpoint:"),
                Select(
                    [],
                    id="judge_model_sep",
                    prompt="Test connection first…",
                    allow_blank=True,
                ),
                id="separate_section",
            ),
            Static(
                "\u26a0\ufe0f Skipping LLM-as-judge means accuracy scores for creative "
                "writing and code explanation will use deterministic grading only. "
                "Results may be less accurate.",
                id="skip_notice",
                classes="warning",
            ),
            Static("", id="status"),
            Horizontal(
                Button("← Back", id="back", variant="default"),
                Button("Next →", id="next", variant="success"),
            ),
            HelpBar("Ctrl+T test · Ctrl+N next · Esc back"),
            id="screen",
        )

    def on_mount(self) -> None:
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        rs = self.query_one("#radio", RadioSet)
        btn = rs.pressed_button
        choice = btn.id if btn else "use_selected"
        self.query_one("#selected_section").display = (choice == "use_selected")
        self.query_one("#separate_section").display = (choice == "use_separate")
        self.query_one("#skip_notice").display = (choice == "skip")
        self.query_one("#test").display = (choice == "use_separate")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:  # type: ignore[override]
        self._refresh_visibility()

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success", "warning": "warning"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    async def action_test_connection(self) -> None:
        rs = self.query_one("#radio", RadioSet)
        btn = rs.pressed_button
        if not btn or btn.id != "use_separate":
            self._refresh_status("Test connection is only available for the separate endpoint option.", "warning")
            return
        judge_url = self.query_one("#judge_url", Input).value.strip()
        judge_key = self.query_one("#judge_key", Input).value.strip()
        if not judge_url:
            self._refresh_status("Enter a separate endpoint URL first.", "error")
            return
        self._refresh_status("Probing separate endpoint…", "info")
        try:
            discovery = await fetch_models(judge_url, judge_key, "auto")
        except Exception as e:  # noqa: BLE001
            self._refresh_status(f"Connection failed: {e}", "error")
            return
        count = len(discovery.models)
        if not discovery.probe_ok:
            self._refresh_status(
                "Connection failed. Check the URL and try again.", "error",
            )
        elif count == 0:
            self._refresh_status(
                "Reachable, but no models found at /models. Check the URL.", "warning",
            )
        else:
            sep = self.query_one("#judge_model_sep", Select)
            sep.set_options([(m.id, m.id) for m in discovery.models])
            sep.allow_blank = False
            self._refresh_status(
                f"Connected. Found {count} model(s). Protocol: {discovery.protocol}.",
                "success",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "test":
            self.run_worker(self.action_test_connection(), exclusive=False)
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
        rs = self.query_one("#radio", RadioSet)
        btn = rs.pressed_button
        choice = btn.id if btn else "use_selected"
        if choice == "skip":
            state.judge_model = None
            state.use_separate_judge = False
        elif choice == "use_separate":
            sel = self.query_one("#judge_model_sep", Select)
            if sel.is_blank():
                self._refresh_status("Test the connection and select a model first.", "error")
                return
            state.use_separate_judge = True
            state.judge_base_url = self.query_one("#judge_url", Input).value.strip()
            state.judge_api_key = self.query_one("#judge_key", Input).value.strip()
            state.judge_model = str(sel.value)
        else:
            sel = self.query_one("#judge_model_sel", Select)
            if not state.selected_models:
                self._refresh_status("No models selected. Go back and pick some models first.", "error")
                return
            if sel.is_blank():
                self._refresh_status("Select a model to use as judge.", "error")
                return
            state.use_separate_judge = False
            state.judge_model = str(sel.value)
        from llm_bench.tui.screens.run import RunScreen

        self.app.push_screen(RunScreen())
