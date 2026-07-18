"""Results dashboard: sortable table + recommendations + export."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static

from llm_bench.results.recommend import recommend
from llm_bench.results.report import markdown_report
from llm_bench.tui.state import AppState
from llm_bench.tui.widgets import HelpBar


class ResultsScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("e", "export", "Export"),
        Binding("escape", "back", "New run"),
    ]

    CSS_PATH = "../styles.tcss"

    def compose(self) -> ComposeResult:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        yield Container(
            Static("6 / 6 — Results", classes="section-title"),
            DataTable(id="table", zebra_stripes=True, cursor_type="row"),
            Static("", id="recs", classes="success"),
            Static("", id="status"),
            Horizontal(
                Button("Export Markdown", id="export_md", variant="primary"),
                Button("Export JSON", id="export_json", variant="default"),
                Button("New run", id="new", variant="success"),
            ),
            HelpBar("E export · Esc new run"),
            id="screen",
        )

    def on_mount(self) -> None:
        state: AppState = self.app.state  # type: ignore[attr-defined]
        if state.run_result is None:
            self._set_status("No results to show.", "error")
            return
        result = state.run_result
        models = sorted({m for m, _ in result.by_model_task.keys()})
        tasks = sorted({t for _, t in result.by_model_task.keys()})
        table = self.query_one("#table", DataTable)
        cols = ["Task / Model", *models]
        table.add_columns(*cols)
        for task in tasks:
            row = [task]
            for m in models:
                score = result.by_model_task.get((m, task))
                if score is None:
                    row.append("—")
                else:
                    row.append(
                        f"{score.pass_rate*100:5.0f}% / {score.mean_tokens_per_s:5.1f}tps / ${score.cost_usd:.4f}"
                    )
            table.add_row(*row)
        # Recommendations
        recs = recommend(result)
        lines = ["**Recommendations:**"]
        for r in recs:
            lines.append(f"  • `{r.category}` → **{r.model or '—'}** ({r.score:.3f}) — {r.detail}")
        self.query_one("#recs", Static).update("\n".join(lines))
        self._set_status(
            f"{len(result.by_model_task)} task/model combinations, {len(result.errors)} errors.",
            "success",
        )

    def _set_status(self, msg: str, kind: str = "info") -> None:
        cls = {"info": "", "error": "error", "success": "success"}[kind]
        w = self.query_one("#status", Static)
        w.set_classes(cls)
        w.update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id == "export_md":
            self._export_md()
        elif event.button.id == "export_json":
            self._export_json()
        elif event.button.id == "new":
            self._new_run()

    def action_export(self) -> None:
        self._export_md()

    def action_back(self) -> None:
        self._new_run()

    def _export_md(self) -> None:
        state: AppState = self.app.state
        if state.run_result is None:
            return
        from datetime import datetime
        from pathlib import Path

        out = Path("./llm_bench_results") / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        markdown_report(state.run_result, out_path=out)
        self._set_status(f"Markdown exported to {out}", "success")

    def _export_json(self) -> None:
        state: AppState = self.app.state
        if state.run_result is None:
            return
        import json
        from dataclasses import asdict
        from datetime import datetime
        from pathlib import Path

        out = Path("./llm_bench_results") / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = state.run_result
        payload = {
            "by_model_task": {
                f"{m}/{t}": asdict(s) for (m, t), s in result.by_model_task.items()
            },
            "errors": [list(e) for e in result.errors],
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        self._set_status(f"JSON exported to {out}", "success")

    def _new_run(self) -> None:
        # Pop all screens back to endpoint
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
