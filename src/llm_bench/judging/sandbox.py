"""Run LLM-generated Python in a sandboxed subprocess."""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: bool
    stdout: str
    stderr: str
    duration_s: float
    detail: str = ""


def _extract_code(output: str) -> str:
    """Extract Python code from a model response.

    Looks for the first ``python`` fenced block, otherwise returns the raw
    output (best-effort) for models that just emit code.
    """
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return textwrap.dedent(fenced.group(1)).strip()
    if "def " in output or "class " in output:
        return output.strip()
    return output.strip()


def run_python(
    code: str,
    test_code: str,
    *,
    timeout_s: float = 10.0,
    mem_mb: int = 256,
) -> ExecResult:
    """Execute ``code + test_code`` in a sandboxed subprocess.

    The subprocess is launched with ``python -I -S -B`` to skip site/usercustomize
    and prevent pyc caching. Resource limits are applied via preexec_fn on POSIX.
    """
    source = _extract_code(code)
    full = source + "\n\n" + textwrap.dedent(test_code)

    def _preexec() -> None:  # pragma: no cover - platform specific
        # Disable core dumps, limit CPU and memory.
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
        try:
            mem_bytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 1, int(timeout_s) + 1))
        except (ValueError, OSError):
            pass

    with tempfile.TemporaryDirectory(prefix="llm_bench_") as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(full)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_preexec if os.name == "posix" else None,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                passed=False,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + f"\n[timeout after {timeout_s}s]",
                duration_s=timeout_s,
                detail="timeout",
            )
        passed = proc.returncode == 0 and "AssertionError" not in proc.stderr
        detail = ""
        if proc.returncode != 0:
            detail = f"exit {proc.returncode}"
        elif "AssertionError" in proc.stderr:
            detail = "assertion failed"
        return ExecResult(
            passed=passed,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=0.0,
            detail=detail,
        )
