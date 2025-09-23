"""
Static analysis utilities (lightweight, no external deps).

Features:
- Recursively scan a directory for Python files
- Report: files scanned, total lines, code/comment/blank lines
- Count functions, classes
- Find TODO/FIXME, and basic import stats

Usage (standalone):
  python static_analysis.py [path]
Defaults to current directory if path not provided.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
import ast
import time
from linting import lint_file_basic
from time_complexity import analyze_time_complexity
from space_complexity import analyze_space_complexity


PYTHON_FILE_PATTERN = re.compile(r"^.*\.py$")
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


@dataclass
class FileStats:
    path: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    function_defs: int
    class_defs: int
    todos: int
    import_lines: int


@dataclass
class Summary:
    files_scanned: int
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    function_defs: int
    class_defs: int
    todos: int
    import_lines: int


def iter_python_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith('.')]
        for fname in filenames:
            if PYTHON_FILE_PATTERN.match(fname):
                yield os.path.join(dirpath, fname)


def analyze_file(path: str) -> FileStats:
    total = code = comment = blank = funcs = classes = todos = imports = 0
    func_def_re = re.compile(r"^\s*def\s+\w+\s*\(")
    class_def_re = re.compile(r"^\s*class\s+\w+\s*[:(]")
    comment_re = re.compile(r"^\s*#")
    import_re = re.compile(r"^\s*(from\s+\w[\w\.]*\s+import\s+|import\s+\w)")
    todo_re = re.compile(r"(?i)\b(TODO|FIXME|HACK|XXX)\b")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                stripped = line.strip()
                if not stripped:
                    blank += 1
                    continue
                if comment_re.match(line):
                    comment += 1
                else:
                    code += 1
                if func_def_re.match(line):
                    funcs += 1
                if class_def_re.match(line):
                    classes += 1
                if import_re.match(line):
                    imports += 1
                if todo_re.search(line):
                    todos += 1
    except OSError as exc:
        # Treat unreadable files as empty; surface a note inline
        print(f"[warn] Could not read {path}: {exc}")

    return FileStats(
        path=path,
        total_lines=total,
        code_lines=code,
        comment_lines=comment,
        blank_lines=blank,
        function_defs=funcs,
        class_defs=classes,
        todos=todos,
        import_lines=imports,
    )


def summarize(stats: List[FileStats]) -> Summary:
    return Summary(
        files_scanned=len(stats),
        total_lines=sum(s.total_lines for s in stats),
        code_lines=sum(s.code_lines for s in stats),
        comment_lines=sum(s.comment_lines for s in stats),
        blank_lines=sum(s.blank_lines for s in stats),
        function_defs=sum(s.function_defs for s in stats),
        class_defs=sum(s.class_defs for s in stats),
        todos=sum(s.todos for s in stats),
        import_lines=sum(s.import_lines for s in stats),
    )


def print_report(root: str, stats: List[FileStats], summary: Summary) -> None:
    print("\n" + "=" * 72)
    print(f"Static Analysis Report: {root}")
    print("=" * 72)
    print(f"Files scanned: {summary.files_scanned}")
    print(f"Lines: total={summary.total_lines}, code={summary.code_lines}, comment={summary.comment_lines}, blank={summary.blank_lines}")
    print(f"Defs: functions={summary.function_defs}, classes={summary.class_defs}")
    print(f"Imports: {summary.import_lines}")
    print(f"TODO/FIXME: {summary.todos}")
    print("-" * 72)
    top_by_lines = sorted(stats, key=lambda s: s.total_lines, reverse=True)[:5]
    if top_by_lines:
        print("Top files by total lines:")
        for s in top_by_lines:
            print(f"  {s.total_lines:6d}  {os.path.relpath(s.path, root)}")
    print("-" * 72)


def _enumerate_files(files: List[str], root: str) -> None:
    for idx, p in enumerate(files, start=1):
        rel = os.path.relpath(p, root)
        print(f"  {idx:3d}. {rel}")


def _prompt_index(max_index: int, prompt: str = "Select #: ") -> Optional[int]:
    try:
        val = input(prompt).strip()
        if val == "":
            return None
        idx = int(val)
        if 1 <= idx <= max_index:
            return idx
        print("Invalid index.")
        return None
    except ValueError:
        print("Please enter a number.")
        return None


def _read_file_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except OSError as exc:
        return [f"[error] Could not read {path}: {exc}\n"]


# lint_file_basic is now imported from linting module


def _loop_nesting_depth(node: ast.AST) -> int:
    max_depth = 0
    def visit(n: ast.AST, depth: int) -> None:
        nonlocal max_depth
        if isinstance(n, (ast.For, ast.While, ast.AsyncFor)):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(n):
            visit(child, depth)
    visit(node, 0)
    return max_depth


def _calls_sort_or_sorted(node: ast.AST) -> bool:
    class Finder(ast.NodeVisitor):
        found = False
        def visit_Call(self, call: ast.Call) -> None:  # type: ignore[override]
            func = call.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"sorted", "sort"}:
                self.found = True
            self.generic_visit(call)
    f = Finder()
    f.visit(node)
    return f.found


# analyze_time_complexity is now imported from time_complexity module


def _counts_of_collections(node: ast.AST) -> Tuple[int, int, int, int]:
    lists = dicts = sets = comps = 0
    class Counter(ast.NodeVisitor):
        def visit_List(self, n: ast.List) -> None:  # type: ignore[override]
            nonlocal lists
            lists += 1
            self.generic_visit(n)
        def visit_Dict(self, n: ast.Dict) -> None:  # type: ignore[override]
            nonlocal dicts
            dicts += 1
            self.generic_visit(n)
        def visit_Set(self, n: ast.Set) -> None:  # type: ignore[override]
            nonlocal sets
            sets += 1
            self.generic_visit(n)
        def visit_ListComp(self, n: ast.ListComp) -> None:  # type: ignore[override]
            nonlocal comps
            comps += 1
            self.generic_visit(n)
        def visit_SetComp(self, n: ast.SetComp) -> None:  # type: ignore[override]
            nonlocal comps
            comps += 1
            self.generic_visit(n)
        def visit_DictComp(self, n: ast.DictComp) -> None:  # type: ignore[override]
            nonlocal comps
            comps += 1
            self.generic_visit(n)
    Counter().visit(node)
    return lists, dicts, sets, comps


# analyze_space_complexity is now imported from space_complexity module


def _analysis_menu() -> None:
    print("\nNext actions (on selected file):")
    print("  1. run linting")
    print("  2. find Time complexity")
    print("  3. find spce complexity")
    print("  8. save output to file")
    print("  9. scan new dir - start from scratch")
    print("  0. back to main menu")


def run_static_code_analysis(path: str = ".") -> None:
    root = os.path.abspath(path)
    while True:  # allow rescan (option 9)
        files = list(iter_python_files(root))
        stats = [analyze_file(p) for p in files]
        print_report(root, stats, summarize(stats))
        if not files:
            print("No Python files found.")
            while True:
                print("  9. scan new dir - start from scratch")
                print("  0. back to main menu")
                choice = input("Enter choice: ").strip()
                if choice == "0":
                    return
                if choice == "9":
                    newp = input("Path to analyze (default=.): ").strip() or "."
                    root = os.path.abspath(newp)
                    break
                print("Invalid choice. Try again.")
            continue

        print("Files:")
        _enumerate_files(files, root)

        last_output: List[str] = []
        while True:
            _analysis_menu()
            choice = (input("Enter choice: ").strip() or "")
            if choice == "0":
                return
            if choice == "9":
                # rescan new directory
                newp = input("Path to analyze (default uses previous): ").strip()
                if newp:
                    root = os.path.abspath(newp)
                break  # break inner menu to rescan
            elif choice == "1":
                print("\nRunning basic lint across files...")
                aggregated: List[str] = ["Basic Lint Report"]
                for fp in files:
                    aggregated.append("")
                    aggregated.append(f"== {os.path.relpath(fp, root)} ==")
                    aggregated.extend(lint_file_basic(fp))
                last_output = aggregated
                print("\n".join(aggregated))
            elif choice == "2":
                print("\nHeuristic time complexity (per file)...")
                aggregated: List[str] = ["Time Complexity Report"]
                for fp in files:
                    aggregated.append("")
                    aggregated.append(f"== {os.path.relpath(fp, root)} ==")
                    aggregated.extend(analyze_time_complexity(fp))
                last_output = aggregated
                print("\n".join(aggregated))
            elif choice == "3":
                print("\nHeuristic space complexity (per file)...")
                aggregated: List[str] = ["Space Complexity Report"]
                for fp in files:
                    aggregated.append("")
                    aggregated.append(f"== {os.path.relpath(fp, root)} ==")
                    aggregated.extend(analyze_space_complexity(fp))
                last_output = aggregated
                print("\n".join(aggregated))
            elif choice == "8":
                if not last_output:
                    print("Nothing to save yet. Run an action first.")
                else:
                    ts = int(time.time())
                    out_path = os.path.join(root, f"analysis_{ts}.txt")
                    try:
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(last_output) + "\n")
                        print(f"Saved to {out_path}")
                    except OSError as exc:
                        print(f"Failed to save: {exc}")
            else:
                print("Invalid choice. Try again.")


def main(argv: List[str]) -> None:
    path = argv[1] if len(argv) > 1 else "."
    run_static_code_analysis(path)


if __name__ == "__main__":
    main(sys.argv)


