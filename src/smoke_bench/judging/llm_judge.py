"""LLM-as-judge: score a model output via a second LLM call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from smoke_bench.clients.base import LLMClient
from smoke_bench.judging.deterministic import GradeResult


@dataclass
class JudgeSpec:
    rubric: str
    max_score: float = 5.0
    reference: str | None = None


DEFAULT_RUBRIC = (
    "Rate the response on a scale of 1-5 for overall quality, correctness, "
    "and clarity. Respond with a single JSON object: {\"score\": <number>}"
)


def _parse_score(text: str, max_score: float) -> float:
    """Parse a score from a judge model response. Tolerant of formatting."""
    # Try direct JSON first.
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = obj.get("score")
            if isinstance(v, (int, float)):
                return float(max(0.0, min(max_score, float(v)))) / max_score
        except json.JSONDecodeError:
            pass
    # Fall back to first number in the text.
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if nums:
        try:
            v = float(nums[0])
            return float(max(0.0, min(max_score, v))) / max_score
        except ValueError:
            pass
    return 0.0


async def judge_output(
    *,
    client: LLMClient,
    judge_model: str,
    prompt: str,
    output: str,
    spec: JudgeSpec | None = None,
) -> GradeResult:
    """Score a single output via an LLM judge."""
    from smoke_bench.clients.base import ChatMessage, ChatRequest

    spec = spec or JudgeSpec(rubric=DEFAULT_RUBRIC)
    rubric = spec.rubric
    user_msg = (
        f"## Task\n{prompt}\n\n"
        f"## Model output\n{output}\n\n"
    )
    if spec.reference:
        user_msg += f"## Reference\n{spec.reference}\n\n"
    user_msg += f"## Rubric\n{rubric}\n\nReturn JSON only."
    req = ChatRequest(
        model=judge_model,
        messages=[ChatMessage(role="user", content=user_msg)],
        max_tokens=256,
        temperature=0.0,
        json_mode=True,
    )
    try:
        resp = await client.chat(req)
    except Exception as e:
        return GradeResult(score=0.0, passed=False, detail=f"judge error: {e}")
    score = _parse_score(resp.text, spec.max_score)
    return GradeResult(score=score, passed=score >= 0.6, detail=resp.text[:200])
