"""Static-analysis tests enforcing the project asyncio policy.

All tests are purely static: they parse source files with AST / regex and
never import or execute any application code, so they run instantly and
require no external services.

Policy summary (see AGENTS.md §9):
  APPROVED-1  asyncio.to_thread(blocking_fn, *args)
  APPROVED-2  asyncio.gather(*coroutines)          — within-node concurrent HTTP/LLM
  APPROVED-3  asyncio.wait_for(single_call, N)     — SDK without native async timeout
  APPROVED-4  Infrastructure only (asyncio.run, Task, Queue, sleep, CancelledError)
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MAIN_PY = PROJECT_ROOT / "main.py"
API_PY = SRC_DIR / "api.py"

_UNAPPROVED_PATTERNS = [
    # asyncio.Semaphore is banned
    r"asyncio\.Semaphore\b",
    # asyncio.Lock is banned (use LangGraph max_concurrency)
    r"asyncio\.Lock\b",
    # asyncio.Event is banned
    r"asyncio\.Event\b",
    # asyncio.BoundedSemaphore is banned
    r"asyncio\.BoundedSemaphore\b",
]


def _all_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _has_approved_comment_before(lines: list[str], lineno: int) -> bool:
    """Return True if the line immediately before lineno contains asyncio-APPROVED."""
    if lineno <= 1:
        return False
    prev = lines[lineno - 2].strip()  # lineno is 1-indexed
    return "asyncio-APPROVED" in prev


# ---------------------------------------------------------------------------
# Test 1 — no unapproved asyncio primitives anywhere in src/
# ---------------------------------------------------------------------------


def test_no_unapproved_asyncio_calls():
    """No asyncio.Semaphore / Lock / Event / BoundedSemaphore in src/ or scripts/."""
    violations: list[str] = []
    for path in _all_py_files(SRC_DIR) + _all_py_files(SCRIPTS_DIR) + [MAIN_PY]:
        text = path.read_text(encoding="utf-8")
        for pattern in _UNAPPROVED_PATTERNS:
            for m in re.finditer(pattern, text):
                lineno = text[: m.start()].count("\n") + 1
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {m.group()}")
    assert not violations, "Unapproved asyncio primitives found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Test 2 — no asyncio.Semaphore (belt-and-suspenders; also caught by test 1)
# ---------------------------------------------------------------------------


def test_no_asyncio_semaphore():
    """asyncio.Semaphore must not appear anywhere — use max_concurrency on compile()."""
    violations: list[str] = []
    for path in _all_py_files(SRC_DIR) + _all_py_files(SCRIPTS_DIR) + [MAIN_PY]:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"asyncio\.Semaphore\b", text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
    assert not violations, "asyncio.Semaphore found (banned):\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Test 3 — asyncio.create_task only in src/api.py
# ---------------------------------------------------------------------------


def test_no_asyncio_create_task_outside_api():
    """asyncio.create_task must only appear in src/api.py."""
    violations: list[str] = []
    for path in _all_py_files(SRC_DIR) + _all_py_files(SCRIPTS_DIR) + [MAIN_PY]:
        if path == API_PY:
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"asyncio\.create_task\b", text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
    assert not violations, (
        "asyncio.create_task outside src/api.py (must use LangGraph Send() instead):\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 4 — asyncio.Queue only in src/api.py
# ---------------------------------------------------------------------------


def test_no_asyncio_queue_outside_api():
    """asyncio.Queue must only appear in src/api.py (SSE infrastructure)."""
    violations: list[str] = []
    for path in _all_py_files(SRC_DIR) + _all_py_files(SCRIPTS_DIR) + [MAIN_PY]:
        if path == API_PY:
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"asyncio\.Queue\b", text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
    assert not violations, (
        "asyncio.Queue outside src/api.py:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 5 — CancelledError is always re-raised in src/api.py
# ---------------------------------------------------------------------------


def test_cancelled_error_is_reraised_in_api():
    """Every `except asyncio.CancelledError` block in api.py must re-raise."""
    source = API_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Check if this handler catches asyncio.CancelledError
        t = node.type
        if t is None:
            continue
        name = ""
        if isinstance(t, ast.Attribute):
            name = f"{getattr(t.value, 'id', '')}.{t.attr}"
        elif isinstance(t, ast.Name):
            name = t.id
        if name != "asyncio.CancelledError":
            continue

        # Check that the handler body contains a bare `raise`
        has_reraise = any(
            isinstance(stmt, ast.Raise) and stmt.exc is None
            for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        )
        if not has_reraise:
            violations.append(f"line {node.lineno}: CancelledError handler missing bare raise")

    assert not violations, (
        "CancelledError handlers in api.py that do not re-raise:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 6 — deep_web_graph uses wait_for pattern for primary call
# ---------------------------------------------------------------------------


def test_deep_web_primary_timeout_pattern():
    """deep_web_try_primary must use asyncio.wait_for (APPROVED-3)."""
    # Canonical implementation moved to subgraphs/deep_web.py; agents/ shim re-exports
    path = SRC_DIR / "graph" / "subgraphs" / "deep_web.py"
    source = path.read_text(encoding="utf-8")
    assert "asyncio.wait_for" in source, (
        "deep_web.py: deep_web_try_primary must use asyncio.wait_for (APPROVED-3)"
    )
    assert "asyncio-APPROVED-3" in source, (
        "deep_web.py: asyncio.wait_for call must be annotated with asyncio-APPROVED-3"
    )


# ---------------------------------------------------------------------------
# Test 7 — all LLM graph nodes have state guards
# ---------------------------------------------------------------------------


def test_all_llm_nodes_have_state_guards():
    """Key LLM worker nodes must contain state guards (skip-if-already-computed)."""
    checks = [
        (
            SRC_DIR / "graph" / "agents" / "slr_graph.py",
            "slr_fetch_source",
            'state.get("result")',
        ),
        (
            SRC_DIR / "graph" / "agents" / "lbd_graph.py",
            "lbd_fetch_concept_papers",
            'state.get("result")',
        ),
        (
            SRC_DIR / "graph" / "agents" / "deep_web_graph.py",
            "deep_web_search_round",
            'state.get("result")',
        ),
        (
            SRC_DIR / "graph" / "agents" / "edison_graph.py",
            "edison_search",
            'state.get("papers")',
        ),
    ]
    violations: list[str] = []
    for path, fn_name, guard_expr in checks:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != fn_name:
                continue
            fn_src = ast.get_source_segment(source, node) or ""
            if guard_expr not in fn_src:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}::{fn_name}: "
                    f"missing state guard `{guard_expr}`"
                )
    assert not violations, (
        "LLM/HTTP worker nodes missing state guards:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 8 — mock inputs trigger correct guards (integration smoke)
# ---------------------------------------------------------------------------


def test_mock_inputs_trigger_correct_guards():
    """Fixture JSON files exist for each agent graph and contain the guard-triggering key."""
    import json

    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures" / "agent_mock_inputs"

    checks = [
        ("slr_agent_mock_input.json", "result"),
        ("lbd_agent_mock_input.json", "result"),
        ("deep_web_agent_mock_input.json", "result"),
        ("edison_agent_mock_input.json", "papers"),
    ]
    violations: list[str] = []
    for filename, key in checks:
        fixture_path = fixtures_dir / filename
        if not fixture_path.exists():
            violations.append(f"Missing fixture: {fixture_path.relative_to(PROJECT_ROOT)}")
            continue
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        if key not in data:
            violations.append(
                f"{fixture_path.relative_to(PROJECT_ROOT)}: missing key '{key}'"
            )
    assert not violations, "Fixture problems:\n" + "\n".join(violations)
