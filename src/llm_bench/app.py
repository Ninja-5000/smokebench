"""Textual application entry point."""

from __future__ import annotations

import argparse
import sys

from textual.app import App

from llm_bench.config import CONFIG_PATH, load_config, save_config
from llm_bench.tui.screens.endpoint import EndpointScreen
from llm_bench.tui.state import AppState


class LLMBenchApp(App):
    """Top-level Textual app."""

    TITLE = "llm-bench"
    SUB_TITLE = "LLM benchmarking TUI"

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state

    def on_mount(self) -> None:
        self.push_screen(EndpointScreen())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm-bench", description=__doc__)
    parser.add_argument("--no-tui", action="store_true", help="Reserved for non-TUI use.")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """Launch the TUI. Returns a process exit code."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config()
    state = AppState(
        base_url=cfg.endpoint.base_url,
        api_key=cfg.endpoint.api_key.get_secret_value(),
        protocol=cfg.endpoint.protocol,
        protocol_detected=cfg.endpoint.protocol,
        selected_models=list(cfg.last_models),
        selected_benchmarks=list(cfg.last_benchmarks),
        sample_overrides=dict(cfg.sample_overrides),
        pricing=cfg.pricing,
    )
    app = LLMBenchApp(state)
    try:
        app.run()
    finally:
        # Persist a few choices for next time.
        cfg.endpoint.base_url = state.base_url or cfg.endpoint.base_url
        cfg.endpoint.api_key.set_secret_value(state.api_key) if state.api_key else None
        cfg.endpoint.protocol = state.protocol or cfg.endpoint.protocol
        cfg.last_models = list(state.selected_models)
        cfg.last_benchmarks = list(state.selected_benchmarks)
        cfg.sample_overrides = dict(state.sample_overrides)
        save_config(cfg, CONFIG_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
