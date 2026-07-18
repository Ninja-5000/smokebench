"""Endpoint configuration screen."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static

from llm_bench.clients.detect import detect_protocol
from llm_bench.discovery import fetch_models
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


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
            HelpBar("Ctrl+T test · Ctrl+N next · Q quit"),
            id="screen",
        )

    def on_mount(self) -> None:
        self.query_one("#base_url", Input).focus()

    def _gather(self) -> tuple[str, str, str]:
        return (
            self.query_one("#base_url", Input).value.strip(),
            self.query_one("#api_key", Input).value.strip(),
            self.query_one("#protocol", Select).value or "auto",
        )

    def _set_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        widget = self.query_one("#status", Static)
        widget.set_classes(cls)
        widget.update(msg)

    async def action_test_connection(self) -> None:
        base_url, api_key, protocol = self._gather()
        if not base_url:
            self._set_status("Base URL is required.", "error")
            return
        self._set_status("Probing endpoint…", "info")
        try:
            chosen = await detect_protocol(base_url, api_key) if protocol == "auto" else protocol
            discovery = await fetch_models(base_url, api_key, protocol)
            count = len(discovery.models)
            self._set_status(
                f"Connected. Detected protocol: {chosen}. Found {count} models.",
                "success",
            )
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Connection failed: {e}", "error")

    def action_next_screen(self) -> None:
        self._advance()

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "test":
            self.run_worker(self.action_test_connection(), exclusive=False)
        elif event.button.id == "next":
            self._advance()

    def _advance(self) -> None:
        base_url, api_key, protocol = self._gather()
        if not base_url:
            self._set_status("Base URL is required.", "error")
            return
        state: AppState = self.app.state  # type: ignore[attr-defined]
        state.base_url = base_url
        state.api_key = api_key
        state.protocol = protocol
        from llm_bench.tui.screens.models import ModelsScreen

        self.app.push_screen(ModelsScreen())
