"""JSON / JSONL reader used by benchmark dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]
