"""Advanced options: per-benchmark sample counts, token limits, and global settings."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static

from smoke_bench.benchmarks import ALL_BENCHMARKS
from smoke_bench.tui.state import AppState
from smoke_bench.tui.widgets import HelpBar


class AdvancedScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "back", "Back"),
    ]

    CSS_PATH = "../styles.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._defaults: dict[str, tuple[int, int]] = {}
        self._orig_concurrency: int = 4
        self._orig_global_tokens: int = 4096

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Advanced settings", classes="section-title"),
            Static(
                "Override sample counts and output token limits per benchmark, "
                "or change global defaults below. Values are pre-filled with defaults.",
                classes="help",
            ),
            VerticalScroll(
                Container(id="advanced-grid", classes="advanced-grid"),
                Horizontal(
                    Button("Reset to defaults", id="reset_all", variant="default"),
                    Button("Save", id="save", variant="success"),
                    Button("Cancel", id="cancel", variant="default"),
                    classes="action-row",
                ),
                id="advanced-scroll",
            ),
            Static("", id="status"),
            HelpBar("Esc back"),
            id="screen",
        )

    def on_mount(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        grid = self.query_one("#advanced-grid", Container)

        self._orig_concurrency = state.max_concurrency
        self._orig_global_tokens = state.global_max_tokens

        children: list = [
            Static("Benchmarks", classes="full-row subsection-title"),
            Label("Benchmark", classes="grid-header"),
            Label("Samples", classes="grid-header"),
            Label("Max Tokens", classes="grid-header"),
        ]

        for benchmark_cls in ALL_BENCHMARKS:
            inst = benchmark_cls()
            name = inst.name
            samples = list(inst.samples)
            default_n = len(samples)
            default_t = (
                samples[0].request_kwargs.get("max_tokens", 4096)
                if samples
                else 4096
            )
            self._defaults[name] = (default_n, default_t)

            n_cur = state.sample_overrides.get(name, default_n)
            t_cur = state.token_overrides.get(name, default_t)

            children.append(Label(f"{name}"))
            children.append(
                Horizontal(
                    Input(value=str(n_cur), id=f"n_{name}", classes="field-input"),
                    Button("\u21ba", id=f"reset_n_{name}", classes="reset-inside"),
                    classes="input-wrap",
                )
            )
            children.append(
                Horizontal(
                    Input(value=str(t_cur), id=f"t_{name}", classes="field-input"),
                    Button("\u21ba", id=f"reset_t_{name}", classes="reset-inside"),
                    classes="input-wrap",
                )
            )

        children.append(Static("Global settings", classes="full-row subsection-title"))
        children.append(Static("\u2500" * 60, classes="full-row"))

        children.append(Label("Concurrency:", classes="global-label"))
        children.append(Static(""))
        children.append(
            Horizontal(
                Input(value=str(state.max_concurrency), id="concurrency", classes="field-input"),
                Button("\u21ba", id="reset_concurrency", classes="reset-inside"),
                classes="input-wrap",
            )
        )

        children.append(Label("Global default max tokens:", classes="global-label"))
        children.append(Static(""))
        children.append(
            Horizontal(
                Input(value=str(state.global_max_tokens), id="global_max_tokens", classes="field-input"),
                Button("\u21ba", id="reset_global_max_tokens", classes="reset-inside"),
                classes="input-wrap",
            )
        )

        grid.mount(*children)

        for name, (default_n, default_t) in self._defaults.items():
            self._sync_reset(f"n_{name}", default_n)
            self._sync_reset(f"t_{name}", default_t)
        self._sync_reset("concurrency", self._orig_concurrency)
        self._sync_reset("global_max_tokens", self._orig_global_tokens)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def _sync_reset(self, input_id: str, default_val: int) -> None:
        try:
            inp = self.query_one(f"#{input_id}", Input)
            btn = self.query_one(f"#reset_{input_id}", Button)
            btn.visible = inp.value.strip() != str(default_val)
        except Exception:
            pass

    def _reset_input(self, input_id: str) -> None:
        inp = self.query_one(f"#{input_id}", Input)
        if input_id == "concurrency":
            inp.value = str(self._orig_concurrency)
        elif input_id == "global_max_tokens":
            inp.value = str(self._orig_global_tokens)
        elif input_id.startswith("n_"):
            name = input_id[2:]
            default_n, _ = self._defaults.get(name, (0, 0))
            inp.value = str(default_n)
        elif input_id.startswith("t_"):
            name = input_id[2:]
            _, default_t = self._defaults.get(name, (0, 0))
            inp.value = str(default_t)
        self._sync_reset(input_id, int(inp.value))

    def _reset_all_to_defaults(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        state.sample_overrides.clear()
        state.token_overrides.clear()

        for name, (default_n, default_t) in self._defaults.items():
            self.query_one(f"#n_{name}", Input).value = str(default_n)
            self.query_one(f"#t_{name}", Input).value = str(default_t)

        state.max_concurrency = 4
        self.query_one("#concurrency", Input).value = "4"

        state.global_max_tokens = 4096
        self.query_one("#global_max_tokens", Input).value = "4096"

        for name, (default_n, default_t) in self._defaults.items():
            self._sync_reset(f"n_{name}", default_n)
            self._sync_reset(f"t_{name}", default_t)
        self._sync_reset("concurrency", 4)
        self._sync_reset("global_max_tokens", 4096)

        self._refresh_status("Reset to defaults.", "success")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _refresh_status(self, msg: str, kind: str = "info") -> None:
        cls_map = {"info": "", "error": "error", "success": "success"}
        w = self.query_one("#status", Static)
        w.set_classes(cls_map.get(kind, ""))
        w.update(msg)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id or ""
        if input_id.startswith("n_"):
            name = input_id[2:]
            default, _ = self._defaults.get(name, (0, 0))
            self._sync_reset(input_id, default)
        elif input_id.startswith("t_"):
            name = input_id[2:]
            _, default = self._defaults.get(name, (0, 0))
            self._sync_reset(input_id, default)
        elif input_id == "concurrency":
            self._sync_reset(input_id, self._orig_concurrency)
        elif input_id == "global_max_tokens":
            self._sync_reset(input_id, self._orig_global_tokens)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        btn_id = event.button.id or ""

        if btn_id == "save":
            state: AppState = self.app.state  # type: ignore[attr-defined]
            state.sample_overrides.clear()
            state.token_overrides.clear()

            for benchmark_cls in ALL_BENCHMARKS:
                name = benchmark_cls().name
                default_n, default_t = self._defaults.get(name, (0, 0))

                raw_n = self.query_one(f"#n_{name}", Input).value.strip()
                if raw_n:
                    try:
                        val = int(raw_n)
                        if val != default_n:
                            state.sample_overrides[name] = val
                    except ValueError:
                        self._refresh_status(
                            f"{name}: invalid sample count '{raw_n}'", "error"
                        )
                        return

                raw_t = self.query_one(f"#t_{name}", Input).value.strip()
                if raw_t:
                    try:
                        val = int(raw_t)
                        if val != default_t:
                            state.token_overrides[name] = val
                    except ValueError:
                        self._refresh_status(
                            f"{name}: invalid max tokens '{raw_t}'", "error"
                        )
                        return

            concurrency_raw = self.query_one("#concurrency", Input).value.strip()
            try:
                state.max_concurrency = (
                    max(1, int(concurrency_raw)) if concurrency_raw else 4
                )
            except ValueError:
                state.max_concurrency = 4

            global_tokens_raw = (
                self.query_one("#global_max_tokens", Input).value.strip()
            )
            if global_tokens_raw:
                try:
                    state.global_max_tokens = int(global_tokens_raw)
                except ValueError:
                    self._refresh_status(
                        f"Invalid global max tokens '{global_tokens_raw}'", "error"
                    )
                    return

            self._refresh_status("Saved.", "success")
            self.app.pop_screen()

        elif btn_id == "cancel":
            self.app.pop_screen()

        elif btn_id == "reset_all":
            self._reset_all_to_defaults()

        elif btn_id.startswith("reset_"):
            target = btn_id[len("reset_"):]
            self._reset_input(target)

    def action_back(self) -> None:
        self.app.pop_screen()
