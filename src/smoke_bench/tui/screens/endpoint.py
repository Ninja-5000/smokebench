"""Endpoint configuration screen."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static

from smoke_bench.discovery import fetch_models
from smoke_bench.tui.state import AppState
from smoke_bench.tui.widgets import HelpBar


class EndpointScreen(Screen):
    """Collect base URL, API key, and protocol."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+t", "test_connection", "Test connection"),
        Binding("ctrl+n", "next_screen", "Next →", priority=True),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        yield Container(
            Static("1 / 6 — Endpoint", classes="section-title"),
            Grid(
                Label("Base URL"),
                Input(
                    value=state.base_url or "https://api.openai.com/v1",
                    id="base_url",
                    placeholder="https://api.openai.com/v1",
                ),
                Label("API key"),
                Input(
                    value=state.api_key,
                    id="api_key",
                    password=True,
                    placeholder="sk-...",
                ),
                Label("Protocol"),
                Select(
                    [
                        ("Auto-detect", "auto"),
                        ("OpenAI-compatible", "openai"),
                        ("Anthropic-compatible", "anthropic"),
                    ],
                    value=state.protocol or "auto",
                    id="protocol",
                    allow_blank=False,
                ),
                id="form",
                classes="endpoint-grid",
            ),
            Static("", id="status"),
            Horizontal(
                Button("Test connection", id="test", variant="primary"),
                Button("Next →", id="next", variant="success"),
                id="actions",
            ),
            HelpBar("Ctrl+T test · Ctrl+N next · Ctrl+Q quit"),
            id="screen",
        )

    def on_mount(self) -> None:
        self.query_one("#base_url", Input).focus()

    def _gather(self) -> tuple[str, str, str]:
        protocol_value = self.query_one("#protocol", Select).value
        return (
            self.query_one("#base_url", Input).value.strip(),
            self.query_one("#api_key", Input).value.strip(),
            protocol_value if isinstance(protocol_value, str) else "auto",
        )

    def _set_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success", "warning": "warning"}[kind]
        widget = self.query_one("#status", Static)
        widget.set_classes(cls)
        widget.update(msg)

    async def _probe(self) -> tuple[object | None, tuple[str, str, str] | None]:
        """Test the endpoint connection and return the discovery result with form values."""
        values = self._gather()
        base_url, api_key, protocol = values
        if not base_url:
            self._set_status("Base URL is required.", "error")
            return None, values
        self._set_status("Probing endpoint…", "info")
        try:
            discovery = await fetch_models(base_url, api_key, protocol)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Connection failed: {e}", "error")
            return None, values
        count = len(discovery.models)
        chosen = discovery.protocol
        if not discovery.probe_ok:
            self._set_status(
                "Connection failed. Check the URL and try again.", "error",
            )
            return None, values
        if count == 0:
            self._set_status(
                f"Reachable (protocol: {chosen}), but no models found at /models. "
                "Check the URL or add models manually on the next screen.",
                "warning",
            )
            return discovery, values
        self._set_status(
            f"Connected. Detected protocol: {chosen}. Found {count} model(s).",
            "success",
        )
        return discovery, values

    async def action_test_connection(self) -> None:
        await self._probe()

    def action_next_screen(self) -> None:
        self.run_worker(self._advance(), exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "test":
            self.run_worker(self.action_test_connection(), exclusive=False)
        elif event.button.id == "next":
            self.run_worker(self._advance(), exclusive=True)

    async def _advance(self) -> None:
        discovery, values = await self._probe()
        if discovery is None:
            return
        state: AppState = self.app.state  # type: ignore[attr-defined]
        if values is not None:
            state.base_url, state.api_key, state.protocol = values
        from smoke_bench.tui.screens.models import ModelsScreen

        self.app.push_screen(ModelsScreen())
