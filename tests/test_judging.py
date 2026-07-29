"""Tests for deterministic and LLM graders."""

from __future__ import annotations

from smoke_bench.judging import (
    grade_contains,
    grade_cosine_sim,
    grade_exact,
    grade_json_schema,
    grade_numeric,
    grade_regex,
    grade_rouge_l,
    run_python,
)


def test_grade_exact() -> None:
    assert grade_exact("42", "42").passed
    assert grade_exact("42 ", "42", case_sensitive=False).passed
    assert not grade_exact("43", "42").passed


def test_grade_regex() -> None:
    res = grade_regex("The answer is 42", r"answer is (?P<answer>\d+)")
    assert res.passed
    assert res.detail == "42"


def test_grade_contains() -> None:
    assert grade_contains("hello world", "world").passed
    assert not grade_contains("hello", "world").passed


def test_grade_numeric() -> None:
    assert grade_numeric("The answer is 42", 42).passed
    assert not grade_numeric("no number here", 42).passed
    assert grade_numeric("Answer: 3.1415", 3.14, tolerance=0.01).passed


def test_grade_cosine_sim() -> None:
    res = grade_cosine_sim("the quick brown fox", "the quick brown dog")
    assert 0.5 < res.score < 1.0


def test_grade_rouge_l() -> None:
    res = grade_rouge_l(
        "the cat sat on the mat",
        "the cat is on the mat",
    )
    assert 0.5 < res.score <= 1.0


def test_grade_json_schema_pass() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    res = grade_json_schema('{"name": "Alice", "age": 30}', schema)
    assert res.passed


def test_grade_json_schema_fail_missing_field() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    assert not grade_json_schema('{"y": 1}', schema).passed


def test_grade_json_schema_with_fence() -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    out = "Here you go:\n```json\n{\"ok\": true}\n```"
    assert grade_json_schema(out, schema).passed


def test_run_python_passing() -> None:
    code = "def add(a, b):\n    return a + b\n"
    test = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    res = run_python(code, test, timeout_s=5.0)
    assert res.passed, res.stderr


def test_run_python_failing() -> None:
    code = "def add(a, b):\n    return a - b\n"
    test = "assert add(1, 2) == 3\n"
    res = run_python(code, test, timeout_s=5.0)
    assert not res.passed


def test_run_python_extracts_fenced() -> None:
    code = "Here is the code:\n```python\ndef f(): return 1\n```\nHope that helps."
    test = "assert f() == 1\n"
    res = run_python(code, test, timeout_s=5.0)
    assert res.passed, res.stderr
