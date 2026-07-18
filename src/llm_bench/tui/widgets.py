"""Reusable Textual widgets."""

from __future__ import annotations

from textual.widgets import Static


class HelpBar(Static):
    """A small status/help line shown at the bottom of screens."""

    DEFAULT_CSS = """
    HelpBar {
        height: 1;
        dock: bottom;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    """
