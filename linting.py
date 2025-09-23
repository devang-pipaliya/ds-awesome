from __future__ import annotations

from typing import List


def _read_file_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except OSError as exc:
        return [f"[error] Could not read {path}: {exc}\n"]


def lint_file_basic(path: str, max_len: int = 100) -> List[str]:
    issues: List[str] = []
    lines = _read_file_lines(path)
    for i, line in enumerate(lines, start=1):
        if len(line.rstrip("\n")) > max_len:
            issues.append(f"L{i}: line too long (> {max_len})")
        if line.rstrip("\n").endswith(" ") or "\t" in line:
            if line.rstrip("\n").endswith(" "):
                issues.append(f"L{i}: trailing whitespace")
            if "\t" in line:
                issues.append(f"L{i}: tab character found (use spaces)")
    if lines and not lines[-1].endswith("\n"):
        issues.append("EOF: no newline at end of file")
    return issues or ["No basic lint issues found."]


