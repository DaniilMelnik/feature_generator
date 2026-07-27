"""Static (pre-execution) safety checks for LLM-generated feature modules.

Runs as pure AST analysis -- no code execution -- so it is safe to run on
fully untrusted source before it is ever handed to the Docker sandbox. This
is tier 1 of the two-tier defense described in the design; Docker (tier 2)
is defense-in-depth for whatever this can't catch (e.g. resource abuse,
infinite loops, or obfuscation this allowlist-based analysis misses).

Default-deny: any import whose root module isn't explicitly allowlisted is
rejected, not just the ones on the denylist. The denylist exists purely to
produce clearer, named violations for the most common dangerous modules.
"""

from __future__ import annotations

import ast

from feature_generator.schemas import StaticCheckResult

ALLOWED_MODULES = frozenset(
    {
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "math",
        "datetime",
        "re",
        "collections",
        "itertools",
        "functools",
        "typing",
    }
)

EXPLICITLY_DENIED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "importlib",
        "ctypes",
        "multiprocessing",
        "pathlib",
        "io",
        "pickle",
        "shelve",
        "sqlite3",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "webbrowser",
        "code",
        "codeop",
        "inspect",
    }
)

DENIED_CALL_NAMES = frozenset({"eval", "exec", "compile", "__import__", "open", "exit", "quit", "input"})

# Common sandbox-escape / introspection vectors (e.g. the classic
# `().__class__.__bases__[0].__subclasses__()` gadget).
DENIED_ATTRS = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__subclasses__",
        "__bases__",
        "__base__",
        "__mro__",
        "__loader__",
        "__code__",
        "__closure__",
        "__reduce__",
        "__reduce_ex__",
        "__getattribute__",
        "__import__",
    }
)


def _root_module(dotted_name: str) -> str:
    return dotted_name.split(".", 1)[0]


def _extract_required_columns_literal(tree: ast.AST) -> list[str] | None:
    """Best-effort extraction of `REQUIRED_COLUMNS = [...]` as a literal list
    of strings. Returns None if not found or not a simple string-list literal.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "REQUIRED_COLUMNS" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List):
            return None
        values: list[str] = []
        for elt in node.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
            else:
                return None
        return values
    return None


def run_static_checks(
    source_code: str,
    *,
    forbidden_target_column: str,
    declared_input_columns: list[str] | None = None,
    allow_target_in_fit: bool = False,
) -> StaticCheckResult:
    """``allow_target_in_fit`` mirrors ``FeatureHypothesis.requires_target_in_fit``:
    when set, target-column references *inside the `fit` function only* are
    permitted (e.g. for target/likelihood encoding) -- `transform` and every
    other function must never reference the target, regardless of this flag,
    since `transform` also runs at inference time when no label exists.
    """
    violations: list[str] = []
    denylist_hits: list[str] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return StaticCheckResult(passed=False, violations=[f"syntax error: {exc}"])

    fit_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit"), None
    )
    fit_descendant_ids = {id(n) for n in ast.walk(fit_node)} if fit_node is not None else set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root in EXPLICITLY_DENIED_MODULES:
                    denylist_hits.append(f"import:{root}")
                    violations.append(f"denied import: '{alias.name}'")
                elif root not in ALLOWED_MODULES:
                    violations.append(f"import not in allowlist: '{alias.name}'")

        elif isinstance(node, ast.ImportFrom):
            root = _root_module(node.module) if node.module else ""
            if root in EXPLICITLY_DENIED_MODULES:
                denylist_hits.append(f"import:{root}")
                violations.append(f"denied import: 'from {node.module} import ...'")
            elif root not in ALLOWED_MODULES:
                violations.append(f"import not in allowlist: 'from {node.module} import ...'")

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DENIED_CALL_NAMES:
                denylist_hits.append(f"call:{func.id}")
                violations.append(f"denied call: '{func.id}(...)'")
            elif isinstance(func, ast.Attribute) and func.attr in DENIED_CALL_NAMES:
                denylist_hits.append(f"call:{func.attr}")
                violations.append(f"denied call: '...{func.attr}(...)'")

        elif isinstance(node, ast.Attribute):
            if node.attr in DENIED_ATTRS:
                denylist_hits.append(f"attr:{node.attr}")
                violations.append(f"denied attribute access: '.{node.attr}'")
            if node.attr == forbidden_target_column:
                exempt = allow_target_in_fit and id(node) in fit_descendant_ids
                if not exempt:
                    denylist_hits.append("target-column-attr")
                    scope = "fit() (requires_target_in_fit was not set)" if allow_target_in_fit else "code"
                    violations.append(
                        f"{scope} references the target column '{forbidden_target_column}' "
                        "as an attribute -- only fit() may read the target, and only when "
                        "requires_target_in_fit=True; transform() must never see it"
                    )

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and forbidden_target_column in node.value:
                exempt = allow_target_in_fit and id(node) in fit_descendant_ids
                if not exempt:
                    denylist_hits.append("target-column-literal")
                    violations.append(
                        f"code contains a string literal referencing the target column "
                        f"'{forbidden_target_column}' outside an allowed fit()-only context -- "
                        "only fit() may read the target, and only when requires_target_in_fit=True; "
                        "transform() must never see it"
                    )

    required_columns = _extract_required_columns_literal(tree)
    if required_columns is None:
        violations.append("REQUIRED_COLUMNS must be a literal list of string constants")
    elif declared_input_columns is not None and set(required_columns) != set(declared_input_columns):
        violations.append(
            f"REQUIRED_COLUMNS {required_columns} does not match the hypothesis's "
            f"declared_input_columns {declared_input_columns}"
        )

    return StaticCheckResult(
        passed=len(violations) == 0,
        violations=violations,
        ast_denylist_hits=denylist_hits,
    )
