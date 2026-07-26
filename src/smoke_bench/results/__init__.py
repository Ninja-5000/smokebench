"""Results: persistence, recommendations, and reports."""

from smoke_bench.results.recommend import CATEGORIES, Recommendation, recommend
from smoke_bench.results.report import markdown_report, terminal_table
from smoke_bench.results.store import save_result

__all__ = [
    "CATEGORIES",
    "Recommendation",
    "markdown_report",
    "recommend",
    "save_result",
    "terminal_table",
]
