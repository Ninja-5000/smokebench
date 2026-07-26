"""Generate a human-readable Markdown report from a ``RunResult``."""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.table import Table

from smoke_bench.results.recommend import recommend
from smoke_bench.runner import RunResult


def markdown_report(result: RunResult, out_path: str | Path | None = None) -> str:
    buf = StringIO()
    lines: list[str] = []
    lines.append("# SmokeBench Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    # Per-task results table
    models = sorted({m for m, _ in result.by_model_task.keys()})
    tasks = sorted({t for _, t in result.by_model_task.keys()})
    lines.append("## Results")
    lines.append("")
    header = ["Task / Model"] + models
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for task in tasks:
        row = [task]
        for m in models:
            score = result.by_model_task.get((m, task))
            if score is None:
                row.append("—")
            else:
                row.append(
                    f"{score.pass_rate*100:.0f}% · {score.mean_score:.2f} · "
                    f"{score.median_latency_s:.2f}s · {score.mean_tokens_per_s:.1f} t/s"
                )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    recs = recommend(result)
    lines.append("| Category | Best model | Aggregate score | Detail |")
    lines.append("|---|---|---|---|")
    for r in recs:
        lines.append(
            f"| {r.category} | {r.model or '—'} | {r.score:.3f} | {r.detail} |"
        )
    lines.append("")

    # Errors
    if result.errors:
        lines.append("## Errors")
        lines.append("")
        for m, t, msg in result.errors:
            lines.append(f"- `{m}` × `{t}`: {msg}")
        lines.append("")

    out = "\n".join(lines)
    if out_path is not None:
        Path(out_path).write_text(out)
    buf.write(out)
    return buf.getvalue()


def terminal_table(result: RunResult) -> str:
    """Render a compact Rich table to a string for the TUI results screen."""
    console = Console(record=True, width=120)
    models = sorted({m for m, _ in result.by_model_task.keys()})
    tasks = sorted({t for _, t in result.by_model_task.keys()})
    table = Table(title="Benchmark results", show_lines=False)
    table.add_column("Task", style="bold")
    for m in models:
        table.add_column(m, justify="right")
    for task in tasks:
        row = [task]
        for m in models:
            score = result.by_model_task.get((m, task))
            if score is None:
                row.append("—")
            else:
                row.append(
                    f"{score.pass_rate*100:5.0f}% / {score.mean_tokens_per_s:5.1f}tps"
                )
        table.add_row(*row)
    console.print(table)

    recs = recommend(result)
    rec_table = Table(title="Recommendations", show_lines=False)
    rec_table.add_column("Category", style="bold cyan")
    rec_table.add_column("Best model", style="green")
    rec_table.add_column("Score", justify="right")
    rec_table.add_column("Detail", style="dim")
    for r in recs:
        rec_table.add_row(
            r.category,
            r.model or "—",
            f"{r.score:.3f}",
            r.detail,
        )
    console.print(rec_table)
    return console.export_text()
