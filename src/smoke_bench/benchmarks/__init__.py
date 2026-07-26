"""Benchmark tasks."""

from smoke_bench.benchmarks.base import (
    Benchmark,
    ProgressCallback,
    Sample,
    SampleResult,
    TaskScore,
)
from smoke_bench.benchmarks.code_explain import CodeExplainBenchmark
from smoke_bench.benchmarks.code_gen import CodeGenBenchmark
from smoke_bench.benchmarks.creative_writing import CreativeWritingBenchmark
from smoke_bench.benchmarks.custom import CustomBenchmark, CustomSpec
from smoke_bench.benchmarks.instruction_json import InstructionJSONBenchmark
from smoke_bench.benchmarks.latency import LatencyBenchmark
from smoke_bench.benchmarks.long_context import LongContextBenchmark
from smoke_bench.benchmarks.math_reason import MathReasonBenchmark
from smoke_bench.benchmarks.summarization import SummarizationBenchmark

ALL_BENCHMARKS: list[type[Benchmark]] = [
    MathReasonBenchmark,
    CodeGenBenchmark,
    CodeExplainBenchmark,
    SummarizationBenchmark,
    InstructionJSONBenchmark,
    LongContextBenchmark,
    CreativeWritingBenchmark,
    LatencyBenchmark,
]


def benchmark_by_name(name: str) -> type[Benchmark] | None:
    for cls in ALL_BENCHMARKS:
        if cls.__name__ == name:
            return cls
        # Try a temporary instance for the ``name`` attribute.
        try:
            if cls().name == name:
                return cls
        except TypeError:
            continue
    return None


def instantiate(name: str, n_samples: int | None = None) -> Benchmark | None:
    cls = benchmark_by_name(name)
    if cls is None:
        return None
    return cls(n_samples=n_samples)


__all__ = [
    "ALL_BENCHMARKS",
    "Benchmark",
    "CodeExplainBenchmark",
    "CodeGenBenchmark",
    "CreativeWritingBenchmark",
    "CustomBenchmark",
    "CustomSpec",
    "InstructionJSONBenchmark",
    "LatencyBenchmark",
    "LongContextBenchmark",
    "MathReasonBenchmark",
    "ProgressCallback",
    "Sample",
    "SampleResult",
    "SummarizationBenchmark",
    "TaskScore",
    "benchmark_by_name",
    "instantiate",
]
