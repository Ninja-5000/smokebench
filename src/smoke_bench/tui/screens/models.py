"""Model selection screen."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, Static

from smoke_bench.discovery import fetch_models, parse_manual_models
from smoke_bench.tui.state import AppState
from smoke_bench.tui.widgets import HelpBar


class ModelsScreen(Screen):
    """Discover and pick models to benchmark."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("space", "toggle_row", "Toggle"),
        Binding("a", "select_all", "Select all"),
        Binding("n", "select_none", "Select none"),
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("2 / 6 — Models", classes="section-title"),
            Static(
                "Discovered models from the endpoint. Use Space to toggle, / to filter.",
                classes="help",
            ),
            Input(placeholder="Filter models…", id="filter"),
            DataTable(id="table", zebra_stripes=True, cursor_type="row"),
            Vertical(
                Label("No models discovered? Enter model ids manually (comma or newline separated):"),
                Input(
                    id="manual",
                    placeholder="gpt-4o-mini, claude-3-5-sonnet, …",
                ),
                Horizontal(
                    Button("Add manual", id="add_manual", variant="default"),
                    Button("Refresh", id="refresh", variant="primary"),
                ),
            ),
            Static("", id="status"),
            Horizontal(
                Button("← Back", id="back", variant="default"),
                Button("Next →", id="next", variant="success"),
                id="actions",
            ),
            HelpBar("Space toggle · A all · N none · Ctrl+N next · Esc back"),
            id="screen",
        )

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("✓", "ID", "Context", "Modalities", "Features", "Source")
        self._refresh_status("Loading…")
        self.run_worker(self._load_models(), exclusive=False)

    async def _load_models(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        try:
            discovery = await fetch_models(
                state.base_url, state.api_key, state.protocol
            )
            state.protocol_detected = discovery.protocol
            state.models = discovery.models
            state.selected_models = [
                mid for mid in state.selected_models
                if mid in {m.id for m in state.models}
            ]
            state.used_fallback = discovery.used_fallback
        except Exception as e:  # noqa: BLE001
            self._refresh_status(f"Failed to fetch models: {e}", "error")
            return
        self._populate_table()
        if not state.models:
            self._refresh_status(
                "No models found. Use the manual entry field below to add ids.",
                "info",
            )
        else:
            self._refresh_status(
                f"Discovered {len(state.models)} model(s) via {state.protocol_detected}.",
                "success",
            )

    def _populate_table(self, filter_text: str = "") -> None:
        state: AppState = self.app.state
        table = self.query_one("#table", DataTable)
        table.clear()
        for m in state.models:
            if filter_text and filter_text.lower() not in m.id.lower():
                continue
            mark = "✓" if m.id in state.selected_models else " "
            table.add_row(
                mark,
                m.id,
                str(m.context_window) if m.context_window else "?",
                ",".join(m.modalities) or "?",
                ",".join(m.features) or "—",
                "auto",
                key=m.id,
            )
        if state.used_fallback:
            self._refresh_status("Using manual entry.", "info")

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_input_changed(self, event: Input.Changed) -> None:  # type: ignore[override]
        if event.input.id == "filter":
            self._populate_table(event.value)

    def action_toggle_row(self) -> None:
        state: AppState = self.app.state
        table = self.query_one("#table", DataTable)
        if table.row_count == 0:
            return
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        except Exception:
            return
        if row_key is None or row_key.value is None:
            return
        model_id = str(row_key.value)
        if model_id in state.selected_models:
            state.selected_models.remove(model_id)
        else:
            state.selected_models.append(model_id)
        self._populate_table(self.query_one("#filter", Input).value)

    def action_select_all(self) -> None:
        state: AppState = self.app.state
        state.selected_models = [m.id for m in state.models]
        self._populate_table(self.query_one("#filter", Input).value)

    def action_select_none(self) -> None:
        state: AppState = self.app.state
        state.selected_models = []
        self._populate_table(self.query_one("#filter", Input).value)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "add_manual":
            spec = self.query_one("#manual", Input).value.strip()
            if not spec:
                return
            new_models = parse_manual_models(spec)
            state: AppState = self.app.state
            existing = {m.id for m in state.models}
            for m in new_models:
                if m.id not in existing:
                    state.models.append(m)
                    existing.add(m.id)
                    state.used_fallback = True
            self._populate_table(self.query_one("#filter", Input).value)
            self._refresh_status(
                f"Added {len(new_models)} manual model(s).", "success"
            )
        elif event.button.id == "refresh":
            self.run_worker(self._load_models(), exclusive=False)
        elif event.button.id == "next":
            self._advance()
        elif event.button.id == "back":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_next_screen(self) -> None:
        self._advance()

    def _advance(self) -> None:
        state: AppState = self.app.state
        if not state.selected_models:
            self._refresh_status("Select at least one model to continue.", "error")
            return
        from smoke_bench.tui.screens.benchmarks import BenchmarksScreen

        self.app.push_screen(BenchmarksScreen())
