from __future__ import annotations

from typing import List, Optional, Set, Tuple, Dict
import ast


def _read_file_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except OSError as exc:
        return [f"[error] Could not read {path}: {exc}\n"]


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


def _detect_logarithmic_while_loops(func: ast.FunctionDef) -> Set[str]:
    # Detect params that get halved inside a while-loop conditionally using broad patterns.
    halved: Set[str] = set()
    halving_ops = (ast.FloorDiv, ast.RShift, ast.Div, ast.Mult)

    def assignment_halves(target_name: str, node: ast.AST) -> bool:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id == target_name:
                if isinstance(node.op, ast.RShift):
                    return True
                if isinstance(node.op, (ast.FloorDiv, ast.Div)) and \
                        isinstance(node.value, ast.Constant) and node.value.value in (2, 2.0):
                    return True
                if isinstance(node.op, ast.Mult) and isinstance(node.value, ast.Constant) \
                        and node.value.value in (0.5,):
                    return True
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == target_name and isinstance(node.value, ast.BinOp):
                if isinstance(node.value.op, halving_ops):
                    left = node.value.left
                    right = node.value.right
                    if isinstance(left, ast.Name) and left.id == target_name and \
                            isinstance(right, ast.Constant) and right.value in (2, 2.0, 0.5):
                        return True
                # Support wrapped expressions like x = (x // 2) + 1
                def contains_halving(expr: ast.AST) -> bool:
                    if isinstance(expr, ast.BinOp) and isinstance(expr.op, halving_ops):
                        if isinstance(expr.left, ast.Name) and expr.left.id == target_name and \
                                isinstance(expr.right, ast.Constant) and expr.right.value in (2, 2.0):
                            return True
                    return any(contains_halving(c) for c in ast.iter_child_nodes(expr))

                if contains_halving(node.value):
                    return True
        return False

    def cond_param_names(test: ast.AST) -> Set[str]:
        names: Set[str] = set()
        for t in ast.walk(test):
            if isinstance(t, ast.Name):
                names.add(t.id)
        return names

    for node in ast.walk(func):
        if isinstance(node, ast.While):
            cond_names = cond_param_names(node.test)
            for name in cond_names:
                for stmt in node.body:
                    if assignment_halves(name, stmt):
                        halved.add(name)
    return halved


def _recursive_call_count(func: ast.FunctionDef) -> int:
    count = 0
    func_name = func.name

    class Finder(ast.NodeVisitor):
        nonlocal_count = 0
        def visit_Call(self, call: ast.Call) -> None:  # type: ignore[override]
            nonlocal count
            target = call.func
            if isinstance(target, ast.Name) and target.id == func_name:
                count += 1
            elif isinstance(target, ast.Attribute) and target.attr == func_name:
                count += 1
            self.generic_visit(call)

    Finder().visit(func)
    return count


def analyze_time_complexity(path: str) -> List[str]:
    try:
        src = "".join(_read_file_lines(path))
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    module_depth = _loop_nesting_depth(tree)
    has_sort_module = _calls_sort_or_sorted(tree)

    lines: List[str] = []
    lines.append(f"Module loop nesting depth: {module_depth}")
    lines.append(f"Module sorting calls: {'yes' if has_sort_module else 'no'}")

    # Per-function assessment
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fname = node.name
            params: Set[str] = {arg.arg for arg in node.args.args}
            degree_map = _data_dependent_loop_degree_map(node, params)
            log_params = _detect_logarithmic_while_loops(node)
            sort_here = _calls_sort_or_sorted(node)
            rec_calls = _recursive_call_count(node)
            rec_class = _estimate_recursion_class(node, params)

            # Base polynomial estimate O(n^k) using data-dependent loop degree
            poly_terms: List[str] = []
            for p, k in sorted(degree_map.items()):
                if k <= 0:
                    continue
                poly_terms.append(p if k == 1 else f"{p}^{k}")
            poly = "O(1)" if not poly_terms else f"O({' * '.join(poly_terms)})"

            parts: List[str] = [poly]
            if log_params:
                terms = [f"log {p}" for p in sorted(log_params)]
                parts.append(f"O({' * '.join(terms)})")
            if sort_here:
                parts.append("O(n log n)")
            if rec_class:
                parts.append(rec_class)

            # Combine heuristically: exponential dominates > polynomial > n log n > log n > 1
            def rank(term: str) -> int:
                if "2^n" in term:
                    return 5
                if "n^log" in term:
                    return 5
                if "n^" in term and "O(n^" in term:
                    return 4
                if "n log n" in term:
                    return 3
                if term == "O(n)":
                    return 2
                if "log n" in term and "n log n" not in term:
                    return 1
                return 0

            parts_unique = []
            for p in parts:
                if p not in parts_unique:
                    parts_unique.append(p)
            parts_sorted = sorted(parts_unique, key=rank, reverse=True)
            combined = " + ".join(parts_sorted) if parts_sorted else "O(1)"

            lines.append("")
            lines.append(f"Function: {fname}")
            lines.append(
                f"  loop degrees: "
                f"{', '.join(f'{p}:{k}' for p, k in sorted(degree_map.items()))}"
            )
            lines.append(
                f"  logarithmic loops: {', '.join(sorted(log_params)) if log_params else 'none'}"
            )
            lines.append(f"  recursive self-calls: {rec_calls}")
            lines.append(f"  sorting calls: {'yes' if sort_here else 'no'}")
            lines.append(f"  estimated: {combined}")

    # Fallback overall if no functions
    if not any(isinstance(n, ast.FunctionDef) for n in tree.body):
        base = "O(1)" if module_depth == 0 else ("O(n)" if module_depth == 1 else f"O(n^{module_depth})")
        if has_sort_module:
            base = f"{base} + O(n log n)"
        lines.append(f"Estimated (module): {base}")

    return lines


def _expr_depends_on_params(expr: ast.AST, params: Set[str]) -> bool:
    class Finder(ast.NodeVisitor):
        found = False
        def visit_Name(self, n: ast.Name) -> None:  # type: ignore[override]
            if n.id in params:
                self.found = True
        def visit_Call(self, call: ast.Call) -> None:  # type: ignore[override]
            # len(param) counts as dependency
            if isinstance(call.func, ast.Name) and call.func.id == 'len':
                for a in call.args:
                    if isinstance(a, ast.Name) and a.id in params:
                        self.found = True
            self.generic_visit(call)
    f = Finder()
    f.visit(expr)
    return f.found


def _data_dependent_loop_degree_map(node: ast.AST, params: Set[str]) -> Dict[str, int]:
    # Track degree per parameter, e.g., for nested loops over n and m -> {'n':1, 'm':1}
    degree: Dict[str, int] = {p: 0 for p in params}

    def dependent_params_in_iter(it: ast.AST) -> Set[str]:
        deps: Set[str] = set()
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range':
            for a in it.args:
                deps |= _extract_param_set(a, params)
        else:
            deps |= _extract_param_set(it, params)
        return deps

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.For):
            deps = dependent_params_in_iter(n.iter)
            for p in deps:
                degree[p] = degree.get(p, 0) + 1
            for c in n.body:
                visit(c)
            for c in n.orelse:
                visit(c)
            return
        if isinstance(n, ast.While):
            deps = _extract_param_set(n.test, params)
            for p in deps:
                degree[p] = degree.get(p, 0) + 1
            for c in n.body:
                visit(c)
            for c in n.orelse:
                visit(c)
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return {p: k for p, k in degree.items() if k > 0}


def _estimate_recursion_class(func: ast.FunctionDef, params: Set[str]) -> Optional[str]:
    # Try to infer T(n) = a T(n/b) + ... or linear/exponential
    a = 0  # branching factor
    b: Optional[float] = None  # shrink factor
    linear_decrement = False

    def analyze_call(call: ast.Call) -> None:
        nonlocal a, b, linear_decrement
        target = call.func
        called = None
        if isinstance(target, ast.Name):
            called = target.id
        elif isinstance(target, ast.Attribute):
            called = target.attr
        if called != func.name:
            return
        a += 1
        # Inspect args to see if any param is reduced: n-1, n//2, n/2
        for arg in call.args:
            # Detect linear decrement n-1, or any wrapped expression subtracting a constant
            if _expr_has_linear_decrement(arg, params):
                linear_decrement = True
            # Detect halving patterns even when wrapped, e.g., helper(n//2 + 1)
            if _expr_has_halving(arg, params):
                b = 2.0

    for n in ast.walk(func):
        if isinstance(n, ast.Call):
            analyze_call(n)

    if a == 0:
        return None
    # Master theorem-like cases
    if b and a >= 1:
        # n^{log_b a}
        return f"O(n^log_{int(b)}({a})) (recursive branching)"
    if a == 1 and (linear_decrement or _detect_logarithmic_while_loops(func)):
        # single recursion on n-1 => O(n); on n/2 => O(log n) (handled by b)
        return "O(n) (single recursive call)"
    if a > 1 and linear_decrement:
        # e.g., Fibonacci-like
        return "O(2^n) (multiple recursive calls)"
    # Fallback
    return "O(n) (recursion heuristic)"


def _extract_param_set(expr: ast.AST, params: Set[str]) -> Set[str]:
    found: Set[str] = set()
    for t in ast.walk(expr):
        if isinstance(t, ast.Name) and t.id in params:
            found.add(t.id)
        if isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id == 'len':
            for a in t.args:
                if isinstance(a, ast.Name) and a.id in params:
                    found.add(a.id)
    return found


def _expr_has_halving(expr: ast.AST, params: Set[str]) -> bool:
    halving_ops = (ast.FloorDiv, ast.RShift, ast.Div)
    for t in ast.walk(expr):
        if isinstance(t, ast.BinOp) and isinstance(t.left, ast.Name) and t.left.id in params:
            if isinstance(t.op, halving_ops) and isinstance(t.right, ast.Constant) \
                    and t.right.value in (2, 2.0):
                return True
    return False


def _expr_has_linear_decrement(expr: ast.AST, params: Set[str]) -> bool:
    for t in ast.walk(expr):
        if isinstance(t, ast.BinOp) and isinstance(t.left, ast.Name) and t.left.id in params:
            if isinstance(t.op, ast.Sub) and isinstance(t.right, ast.Constant):
                return True
    return False


