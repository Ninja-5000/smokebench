"""Custom benchmark: load user-supplied prompts and graders from a YAML/JSON file
or from in-memory spec."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_bench.benchmarks.base import Benchmark, Sample

VALID_GRADERS = {
    "exact",
    "regex",
    "contains",
    "json_schema",
    "rouge_l",
    "numeric",
    "cosine",
    "judge",
    "sandbox",
}


@dataclass
class CustomSpec:
    name: str
    description: str
    samples: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: str | Path) -> "CustomSpec":
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls(
            name=str(data.get("name", "custom")),
            description=str(data.get("description", "")),
            samples=list(data.get("samples", [])),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustomSpec":
        return cls(
            name=str(d.get("name", "custom")),
            description=str(d.get("description", "")),
            samples=list(d.get("samples", [])),
        )


class CustomBenchmark(Benchmark):
    name = "custom"
    description = "User-defined benchmark loaded from a YAML/JSON file or in-memory spec."

    def __init__(self, spec: CustomSpec, n_samples: int | None = None) -> None:
        super().__init__(n_samples)
        self._spec = spec
        self.name = spec.name
        self.description = spec.description
        self._samples_cache: list[Sample] | None = None

    @property
    def samples(self) -> list[Sample]:
        if self._samples_cache is not None:
            return self._samples_cache
        out: list[Sample] = []
        for i, raw in enumerate(self._spec.samples):
            grader = raw.get("grader", "exact")
            if grader not in VALID_GRADERS:
                grader = "exact"
            schema = raw.get("schema")
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    schema = None
            out.append(
                Sample(
                    id=str(raw.get("id", f"{self._spec.name}_{i+1}")),
                    prompt=str(raw.get("prompt", "")),
                    system=raw.get("system"),
                    expected=raw.get("expected"),
                    grader=grader,
                    test_code=raw.get("test_code"),
                    rubric=raw.get("rubric"),
                    reference=raw.get("reference"),
                    schema=schema,
                    request_kwargs=raw.get("request_kwargs", {}),
                    tags=raw.get("tags", {}),
                )
            )
        self._samples_cache = out
        return out
