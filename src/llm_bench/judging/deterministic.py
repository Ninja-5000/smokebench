"""Deterministic graders for LLM outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    from rouge_score import rouge_scorer as _rouge_scorer  # type: ignore
except Exception:  # pragma: no cover - optional dep
    _rouge_scorer = None

try:
    import jsonschema as _jsonschema  # type: ignore
except Exception:  # pragma: no cover
    _jsonschema = None


@dataclass
class GradeResult:
    score: float  # in [0, 1]
    passed: bool
    detail: str = ""


def grade_exact(output: str, expected: str, *, case_sensitive: bool = True) -> GradeResult:
    if not case_sensitive:
        ok = output.strip().lower() == expected.strip().lower()
    else:
        ok = output.strip() == expected.strip()
    return GradeResult(score=1.0 if ok else 0.0, passed=ok)


def grade_regex(output: str, pattern: str) -> GradeResult:
    """Score 1.0 if pattern is found in output, else 0.0.

    Pattern may include a named or numbered capture group ``answer`` whose
    value will be returned in the detail string.
    """
    flags = re.DOTALL
    m = re.search(pattern, output, flags)
    if not m:
        return GradeResult(score=0.0, passed=False, detail="pattern not matched")
    extracted = m.groupdict().get("answer") if m.groupdict() else (m.group(1) if m.groups() else "")
    return GradeResult(score=1.0, passed=True, detail=str(extracted or "").strip())


def grade_contains(output: str, needle: str) -> GradeResult:
    ok = needle in output
    return GradeResult(score=1.0 if ok else 0.0, passed=ok)


def grade_json_schema(output: str, schema: dict[str, Any]) -> GradeResult:
    if _jsonschema is None:
        return GradeResult(score=0.0, passed=False, detail="jsonschema not installed")
    # Try to extract a JSON object from the output.
    text = output.strip()
    candidates: list[str] = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    generic = re.search(r"\{.*\}", output, re.DOTALL)
    if generic:
        candidates.append(generic.group(0))
    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        try:
            _jsonschema.validate(obj, schema)
            return GradeResult(score=1.0, passed=True)
        except _jsonschema.ValidationError as e:
            last_err = e
            continue
    return GradeResult(
        score=0.0,
        passed=False,
        detail=f"schema validation failed: {last_err}",
    )


def grade_rouge_l(output: str, reference: str) -> GradeResult:
    if _rouge_scorer is None:
        # Simple fallback: ROUGE-L F1 using token-level LCS.
        lcs = _lcs_len(output, reference)
        a_tokens = output.split()
        b_tokens = reference.split()
        if not a_tokens or not b_tokens:
            return GradeResult(score=0.0, passed=False)
        precision = lcs / len(a_tokens)
        recall = lcs / len(b_tokens)
        if precision + recall == 0:
            return GradeResult(score=0.0, passed=False)
        f1 = 2 * precision * recall / (precision + recall)
        return GradeResult(score=float(min(1.0, f1)), passed=f1 >= 0.5)
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    s = scorer.score(reference, output)["rougeL"].fmeasure
    return GradeResult(score=float(s), passed=s >= 0.5)


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    a_tokens = a.split()
    b_tokens = b.split()
    m, n = len(a_tokens), len(b_tokens)
    if m * n > 2_000_000:  # safety cap
        return 0
    dp = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return best


def grade_numeric(output: str, expected: float, *, tolerance: float = 1e-3) -> GradeResult:
    """Extract the last number from the output and compare."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", output)
    if not matches:
        return GradeResult(score=0.0, passed=False, detail="no number found")
    try:
        val = float(matches[-1])
    except ValueError:
        return GradeResult(score=0.0, passed=False, detail="unparsable number")
    ok = abs(val - expected) <= tolerance
    return GradeResult(score=1.0 if ok else 0.0, passed=ok, detail=str(val))


def grade_cosine_sim(output: str, reference: str) -> GradeResult:
    """Bag-of-words cosine similarity. Cheap, no extra deps."""
    import math
    from collections import Counter

    def vec(text: str) -> Counter[str]:
        return Counter(re.findall(r"\w+", text.lower()))

    a, b = vec(output), vec(reference)
    if not a or not b:
        return GradeResult(score=0.0, passed=False)
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return GradeResult(score=0.0, passed=False)
    sim = dot / (na * nb)
    return GradeResult(score=sim, passed=sim >= 0.7)
