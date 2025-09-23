from __future__ import annotations

from typing import List, Tuple
import ast


def _read_file_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except OSError as exc:
        return [f"[error] Could not read {path}: {exc}\n"]


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


def analyze_space_complexity(path: str) -> List[str]:
    try:
        src = "".join(_read_file_lines(path))
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    lists, dicts, sets, comps = _counts_of_collections(tree)
    guess = "O(1)"
    if comps > 0 or lists + dicts + sets > 0:
        guess = "O(n) (collections/comprehensions present)"
    details = [
        f"Collections: lists={lists}, dicts={dicts}, sets={sets}, comprehensions={comps}",
        f"Heuristic space complexity: {guess}",
    ]
    return details


