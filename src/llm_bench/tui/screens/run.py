"""Run benchmark with live progress, streaming output, pause/resume."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Log, ProgressBar, Static

from llm_bench.benchmarks import instantiate
from llm_bench.clients.detect import make_client
from llm_bench.results.recommend import recommend
from llm_bench.results.report import markdown_report
from llm_bench.results.store import save_result
from llm_bench.runner import run_all
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


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
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("p", "pause", "Pause/Resume"),
        Binding("c", "cancel", "Cancel"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        yield Container(
            Static("5 / 6 — Running", classes="section-title"),
            Static("", id="summary"),
            Vertical(id="bars"),
            Static("Log:", classes="help"),
            Log(id="log", highlight=False, max_lines=200),
            Static("", id="status"),
            Horizontal(
                Button("Pause", id="pause", variant="warning"),
                Button("Cancel", id="cancel", variant="error"),
            ),
            HelpBar("P pause · C cancel"),
            id="screen",
        )

    def on_mount(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        models = ", ".join(state.selected_models)
        tasks = ", ".join(state.selected_benchmarks)
        self.query_one("#summary", Static).update(
            f"Models: {models}\nBenchmarks: {tasks}"
        )
        # Build progress bars: one per (model, task)
        bars = self.query_one("#bars", Vertical)
        for m in state.selected_models:
            for t in state.selected_benchmarks:
                mid = _sanitize_id(m)
                bars.mount(
                    Static(f"{m} × {t}", id=f"hdr_{mid}_{t}", classes="help")
                )
                bars.mount(
                    ProgressBar(
                        total=1, show_eta=False, id=f"bar_{mid}_{t}"
                    )
                )
        self._paused = False
        self._cancelled = False
        self.run_worker(self._run(), exclusive=True)

    async def _run(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        # Build benchmarks
        benchmarks = []
        for name in state.selected_benchmarks:
            n_override = state.sample_overrides.get(name)
            b = instantiate(name, n_samples=n_override)
            if b is None:
                # Try custom
                for cb in state.custom_benchmarks:
                    if cb.name == name:
                        benchmarks.append(cb)
                        break
            else:
                benchmarks.append(b)
        if not benchmarks:
            self._set_status("No benchmarks to run.", "error")
            return
        # Set totals on progress bars
        for m in state.selected_models:
            for b in benchmarks:
                mid = _sanitize_id(m)
                pb = self.query_one(f"#bar_{mid}_{b.name}", ProgressBar)
                pb.total = max(1, len(b.sliced_samples()))

        def _make_client(model_id: str):
            return make_client(state.protocol_detected or state.protocol, state.base_url, state.api_key)

        def _make_judge_client(model_id: str):
            if state.use_separate_judge and state.judge_base_url:
                return make_client(
                    state.judge_protocol or "openai",
                    state.judge_base_url,
                    state.judge_api_key or "",
                )
            return _make_client(model_id)

        log = self.query_one("#log", Log)

        async def on_event(kind: str, payload: dict) -> None:
            if self._cancelled:
                raise asyncio.CancelledError()
            while self._paused:
                await asyncio.sleep(0.2)
            if kind == "sample_done":
                bar = self.query_one(f"#bar_{_sanitize_id(payload['model'])}_{payload['task']}", ProgressBar)
                bar.progress = payload["completed"]
                log.write_line(
                    f"[{payload['model']}/{payload['task']}] "
                    f"{payload['sample_id']} {'PASS' if payload['passed'] else 'FAIL'} "
                    f"score={payload['score']:.2f} "
                    f"latency={payload['latency_s']:.2f}s "
                    f"tps={payload['tokens_per_s']:.1f}"
                )
            elif kind == "task_done":
                self._set_status(
                    f"Task {payload['task']} done for {payload['model']}.", "info"
                )
            elif kind == "error":
                log.write_line(
                    f"[ERROR] {payload['model']}/{payload['task']}: {payload['msg']}"
                )

        try:
            result = await run_all(
                models=state.selected_models,
                benchmarks=benchmarks,
                make_client=_make_client,
                pricing=state.pricing,
                judge_model=state.judge_model,
                make_judge_client=_make_judge_client,
                on_event=on_event,
                max_concurrency=state.max_concurrency,
            )
        except asyncio.CancelledError:
            self._set_status("Cancelled.", "error")
            return
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Run failed: {e}", "error")
            return
        state.run_result = result
        recs = recommend(result)
        chips = " · ".join(
            f"[b]{r.category}[/b]→{r.model or '—'}" for r in recs if r.model
        )
        log.write_line("")
        log.write_line("=== RECOMMENDATIONS ===")
        log.write_line(chips)
        self._set_status(
            f"Done. {sum(len(v) for v in result.by_model.values())} task/model combinations.",
            "success",
        )

        # Persist
        out_dir = save_result(result, out_dir="./llm_bench_results")
        markdown_report(result, out_path=out_dir / "report.md")
        log.write_line(f"Report saved to {out_dir / 'report.md'}")

        # Push results screen
        from llm_bench.tui.screens.results import ResultsScreen

        self.app.push_screen(ResultsScreen())

    def _set_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "pause":
            self.action_pause()
        elif event.button.id == "cancel":
            self.action_cancel()

    def action_pause(self) -> None:
        self._paused = not self._paused
        self._set_status(
            f"{'Paused' if self._paused else 'Resumed'}.", "info"
        )

    def action_cancel(self) -> None:
        self._cancelled = True
        self._set_status("Cancelling…", "error")
