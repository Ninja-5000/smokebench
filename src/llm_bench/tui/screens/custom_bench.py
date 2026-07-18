"""Add a custom benchmark: a name, description, and a list of samples."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from llm_bench.benchmarks import CustomBenchmark
from llm_bench.benchmarks.custom import CustomSpec
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


PLACEHOLDER = """\
# Custom benchmark: name + samples
name: my_task
description: short description
samples:
  - id: sample_1
    prompt: |
      What is 2+2? Reply with JSON {"answer": <int>}.
    grader: json_schema
    schema:
      type: object
      properties:
        answer: {type: integer}
      required: [answer]
  - id: sample_2
    prompt: "Write a haiku about the moon."
    grader: judge
    rubric: "Score 1-5 for adherence to 5-7-5 syllable structure."
"""


class CustomBenchScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Custom benchmark", classes="section-title"),
            Static(
                "YAML defining `name`, `description`, and `samples`.",
                classes="help",
            ),
            Horizontal(
                Label("Name"),
                Input(placeholder="my_custom_task", id="name"),
            ),
            TextArea.code_editor(PLACEHOLDER, id="yaml", language="yaml"),
            Horizontal(
                Button("Save", id="save", variant="success"),
                Button("Cancel", id="cancel", variant="default"),
            ),
            Static("", id="status"),
            HelpBar("Ctrl+S save · Esc cancel"),
            id="screen",
        )

    def on_mount(self) -> None:
        self.query_one("#name", Input).focus()

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        import yaml

        name = self.query_one("#name", Input).value.strip() or "custom"
        text = self.query_one("#yaml", TextArea).text
        try:
            data = yaml.safe_load(text) or {}
            if "name" not in data:
                data["name"] = name
            spec = CustomSpec.from_dict(data)
            bench = CustomBenchmark(spec)
            state: AppState = self.app.state  # type: ignore[attr-defined]
            state.custom_benchmarks.append(bench)
            self._refresh_status(
                f"Added '{bench.name}' with {len(spec.samples)} sample(s).", "success"
            )
            self.app.pop_screen()
        except Exception as e:  # noqa: BLE001
            self._refresh_status(f"Failed to parse: {e}", "error")
