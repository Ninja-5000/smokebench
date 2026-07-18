"""Results: persistence, recommendations, and reports."""

from llm_bench.results.recommend import CATEGORIES, Recommendation, recommend
from llm_bench.results.report import markdown_report, terminal_table
from llm_bench.results.store import save_result

__all__ = [
    "CATEGORIES",
    "Recommendation",
    "markdown_report",
    "recommend",
    "save_result",
    "terminal_table",
]
