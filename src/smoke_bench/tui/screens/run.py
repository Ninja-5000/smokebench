"""Live benchmark dashboard with per-model progress and sample results."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import Button, Collapsible, DataTable, Log, ProgressBar, Static

from smoke_bench.benchmarks import instantiate
from smoke_bench.benchmarks.base import Benchmark
from smoke_bench.clients.detect import make_client
from smoke_bench.results.recommend import recommend
from smoke_bench.results.report import markdown_report
from smoke_bench.results.store import save_result
from smoke_bench.runner import run_all
from smoke_bench.tui.state import AppState
from smoke_bench.tui.widgets import HelpBar


def _sanitize_id(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch in ("/", ".", ":", " "):
            out.append("-")
        else:
            out.append("_")
    result = "".join(out)
    if result and result[0].isdigit():
        result = "_" + result
    return result


class RunScreen(Screen):
    """Render the current benchmark run as a responsive operations dashboard."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("p", "pause", "Pause/Resume"),
        Binding("c", "cancel", "Cancel"),
    ]
    CSS_PATH = "../styles.tcss"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (80, "-medium"), (120, "-wide")]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("SMOKEBENCH DASHBOARD", classes="dashboard-title"),
            HorizontalScroll(
                Horizontal(
                    Collapsible(
                        VerticalScroll(id="model-progress"),
                        title="PROGRESS",
                        collapsed=False,
                        id="progress-panel",
                        classes="dashboard-panel",
                    ),
                    Vertical(
                        Collapsible(
                            DataTable(
                                id="live-results", zebra_stripes=True, cursor_type="row"
                            ),
                            title="RESULTS",
                            collapsed=False,
                            id="results-panel",
                            classes="dashboard-panel",
                        ),
                        Collapsible(
                            Log(id="log", highlight=False, max_lines=200),
                            title="LOG OUTPUT",
                            collapsed=False,
                            id="log-panel",
                            classes="dashboard-panel",
                        ),
                        id="dashboard-main",
                    ),
                    id="dashboard",
                ),
                id="dashboard-scroll",
            ),
            Horizontal(
                Static("", id="run-status", classes="help"),
                Button("Pause", id="pause", variant="warning"),
                Button("Cancel", id="cancel", variant="error"),
                id="run-actions",
            ),
            HelpBar("P pause · C cancel"),
            id="screen",
        )

    def on_mount(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        self._benchmarks = self._build_benchmarks(state)
        self._runnable = [b for b in self._benchmarks if b.sliced_samples()]
        self._model_totals = {
            model: sum(len(bench.sliced_samples()) for bench in self._runnable)
            for model in state.selected_models
        }
        self._model_status = {model: "Waiting…" for model in state.selected_models}
        self._model_completed = {model: 0 for model in state.selected_models}
        self._completed_samples = 0
        self._total_samples = sum(self._model_totals.values())
        self._paused = False
        self._cancelled = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._layout_mode: str | None = None

        table = self.query_one("#live-results", DataTable)
        table.add_column("SAMPLE ID", width=42)
        table.add_column("STATUS", width=11)
        table.add_column("SCORE", width=10)
        table.add_column("LATENCY", width=12)
        table.add_column("TPS", width=8)

        progress = self.query_one("#model-progress", VerticalScroll)
        for model in state.selected_models:
            mid = _sanitize_id(model)
            progress.mount(
                Static(model, classes="model-name"),
                Static(self._model_status[model], id=f"model_status_{mid}", classes="model-status"),
                ProgressBar(
                    total=max(1, self._model_totals[model]),
                    show_eta=False,
                    id=f"model_bar_{mid}",
                ),
                Static("", classes="model-spacer"),
            )
        for bench in self._benchmarks:
            if not bench.sliced_samples():
                self.query_one("#log", Log).write_line(
                    f"[SKIP] {bench.name} has 0 samples — skipping."
                )
        self._sync_panel_layout(self.size.width)
        self._run_worker = self.run_worker(self._run(), exclusive=True)

    def on_resize(self, event: Resize) -> None:
        self._sync_panel_layout(event.size.width)

    def _sync_panel_layout(self, width: int) -> None:
        """Apply defaults only when the responsive layout mode changes."""
        mode = "narrow" if width < 80 else "desktop"
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        panels = {
            "progress": self.query_one("#progress-panel", Collapsible),
            "results": self.query_one("#results-panel", Collapsible),
            "log": self.query_one("#log-panel", Collapsible),
        }
        if mode == "narrow":
            panels["progress"].collapsed = False
            panels["results"].collapsed = True
            panels["log"].collapsed = True
        else:
            for panel in panels.values():
                panel.collapsed = False

    def _build_benchmarks(self, state: AppState) -> list[Benchmark]:
        benchmarks: list[Benchmark] = []
        for name in state.selected_benchmarks:
            override = state.sample_overrides.get(name)
            token_override = state.token_overrides.get(name)
            benchmark = instantiate(
                name,
                n_samples=override,
                max_tokens_override=token_override,
                global_max_tokens=state.global_max_tokens,
            )
            if benchmark is not None:
                benchmarks.append(benchmark)
                continue
            benchmarks.extend(cb for cb in state.custom_benchmarks if cb.name == name)
        return benchmarks

    def _set_model_status(self, model: str, status: str, kind: str = "") -> None:
        self._model_status[model] = status
        widget = self.query_one(f"#model_status_{_sanitize_id(model)}", Static)
        widget.set_classes(f"model-status {kind}".strip())
        widget.update(status)

    def _set_run_status(self, message: str, kind: str = "") -> None:
        widget = self.query_one("#run-status", Static)
        widget.set_classes(f"help {kind}".strip())
        widget.update(message)

    async def _run(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        if not self._runnable:
            self._set_run_status("No benchmarks to run.", "error")
            return

        def make_benchmark_client(model_id: str):
            return make_client(state.protocol_detected or state.protocol, state.base_url, state.api_key)

        def make_judge_client(model_id: str):
            if state.use_separate_judge and state.judge_base_url:
                return make_client(
                    state.judge_protocol or "openai",
                    state.judge_base_url,
                    state.judge_api_key or "",
                )
            return make_benchmark_client(model_id)

        log = self.query_one("#log", Log)

        async def on_event(kind: str, payload: dict) -> None:
            if self._cancelled:
                raise asyncio.CancelledError()
            if kind == "model_start":
                self._set_model_status(payload["model"], "Starting benchmarks…", "warning")
            elif kind == "task_start":
                self._set_model_status(payload["model"], f"Current: {payload['task']}")
            elif kind == "sample_done":
                self._completed_samples += 1
                self._model_completed[payload["model"]] += 1
                bar = self.query_one(
                    f"#model_bar_{_sanitize_id(payload['model'])}", ProgressBar
                )
                bar.progress = min(self._model_completed[payload["model"]], bar.total)
                self._append_result(payload)
                log.write_line(
                    f"[{payload['model']}/{payload['task']}] {payload['sample_id']} "
                    f"{'PASS' if payload['passed'] else 'FAIL'} score={payload['score']:.2f} "
                    f"latency={payload['latency_s']:.2f}s tps={payload['tokens_per_s']:.1f}"
                )
                if self._paused:
                    self._set_run_status("Paused.", "warning")
                else:
                    self._set_run_status(
                        f"{self._completed_samples} / {self._total_samples} samples complete"
                    )
                if not payload.get("passed") and payload.get("score", 1.0) == 0.0:
                    detail = payload.get("detail", "")
                    if detail:
                        log.write_line(f"  └─ {detail}")
            elif kind == "task_done":
                self._set_model_status(payload["model"], f"Completed: {payload['task']}", "success")
            elif kind == "model_done":
                self._set_model_status(payload["model"], "Finished", "success")
            elif kind == "error":
                self._set_model_status(payload["model"], f"Error: {payload['task']}", "error")
                log.write_line(f"[ERROR] {payload['model']}/{payload['task']}: {payload['msg']}")
            elif kind == "retry":
                log.write_line(
                    f"  ↻ retry {payload['model']}/{payload['sample_id']} "
                    f"attempt {payload['attempt']}: {payload['error']}"
                )

        try:
            result = await run_all(
                models=state.selected_models,
                benchmarks=self._runnable,
                make_client=make_benchmark_client,
                pricing=state.pricing,
                judge_model=state.judge_model,
                make_judge_client=make_judge_client,
                on_event=on_event,
                max_concurrency=state.max_concurrency,
                pause_event=self._pause_event,
            )
        except asyncio.CancelledError:
            self._set_run_status("Cancelled.", "error")
            return
        except Exception as exc:  # noqa: BLE001
            self._set_run_status(f"Run failed: {exc}", "error")
            return

        state.run_result = result
        recommendations = " · ".join(
            f"{item.category}→{item.model or '—'}" for item in recommend(result) if item.model
        )
        log.write_line("=== RECOMMENDATIONS ===")
        log.write_line(recommendations)
        self._set_run_status("Run complete. Saving report…", "success")
        out_dir = save_result(result, out_dir="./smokebench_results")
        markdown_report(result, out_path=out_dir / "report.md")
        log.write_line(f"Report saved to {out_dir / 'report.md'}")

        from smoke_bench.tui.screens.results import ResultsScreen

        self.app.push_screen(ResultsScreen())

    def _append_result(self, payload: dict) -> None:
        status = Text("PASS" if payload["passed"] else "FAIL", style="#a3e635" if payload["passed"] else "#ff6b6b")
        sample = Text.assemble(
            payload["sample_id"], (f" ({payload['model']})", "dim")
        )
        table = self.query_one("#live-results", DataTable)
        table.add_row(
            sample,
            status,
            f"{payload['score']:.2f}",
            f"{payload['latency_s']:.2f}s",
            f"{payload['tokens_per_s']:.1f}",
        )
        table.scroll_end(animate=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "pause":
            self.action_pause()
        elif event.button.id == "cancel":
            self.action_cancel()

    def action_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._pause_event.clear()
            self.query_one("#pause", Button).label = "Resume"
            self._set_run_status("Waiting to pause…", "warning")
        else:
            self._pause_event.set()
            self.query_one("#pause", Button).label = "Pause"
            self._set_run_status("Resumed.", "")

    def action_cancel(self) -> None:
        self._cancelled = True
        self._run_worker.cancel()
        self._set_run_status("Cancelling…", "error")
