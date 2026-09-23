"""
PipelineState contract — the "silently dropped key" bug class.

LangGraph only keeps top-level keys declared on PipelineState (see the long
comment in pipeline/state.py); anything else a node returns just vanishes with
no error. This test scans the node modules' source and fails if a node returns a
key that isn't declared, so a new one can't slip in unnoticed.

KNOWN_UNDECLARED records existing offenders. Read the note on each BEFORE
"fixing" it by declaring the key — some fixes activate code paths that have
never actually run.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

# key -> why it's still undeclared / what declaring it would change
KNOWN_UNDECLARED = {
    "_guard_passed": "read by route_after_draft_guard / route_after_draft_short_guard (default True). Declaring it "
                     "ACTIVATES the guard->redraft loop, which has NO iteration cap — add one first.",
    "_guard_reason": "returned alongside _guard_passed; nothing reads it, safe to declare or drop.",
    "classifier_confidence": "read by route_after_classify (default 'high'), so the low-confidence -> clarify route "
                             "never fires today. Declaring it turns that route on.",
    "clarification_rounds": "clarify_node's round counter always restarts at 1, so any 'max rounds' limit never trips.",
    "clarification_return_to": "clarify_node's return target is ignored (route_after_clarify always falls back to "
                               "'classify'), so clarify->describe never happens.",
}

HELPERS_RETURNING_STATE_UPDATES = {"_finalize_draft", "_apply_escalation_and_confidence"}


def _declared_keys(state_py: pathlib.Path) -> set[str]:
    tree = ast.parse(state_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PipelineState":
            return {s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    raise AssertionError("PipelineState class not found in pipeline/state.py")


def _str_keys(d: ast.Dict) -> set[str]:
    return {k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _own_nodes(func: ast.FunctionDef):
    """
    Every node in `func`'s own body, WITHOUT descending into nested functions, lambdas or classes:
    a helper defined inside a node (describe_node's _request_extra_body returns a request dict) is
    not the node's state update, and its return keys must not be attributed to the node.
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    stack = list(func.body)
    while stack:
        n = stack.pop()
        if isinstance(n, nested):          # checked on pop, so a nested def that is a
            continue                       # top-level statement of the body is skipped too
        yield n
        stack.extend(ast.iter_child_nodes(n))


def _returned_keys(func: ast.FunctionDef) -> dict[str, int]:
    """String keys of dict literals that flow into a `return` (directly, inside a tuple, or via a local name)."""
    local: dict[str, set[str]] = {}
    for n in _own_nodes(func):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    local.setdefault(t.id, set()).update(_str_keys(n.value))
    for n in _own_nodes(func):                                 # result["key"] = ...
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in local
                        and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)):
                    local[t.value.id].add(t.slice.value)
    found: dict[str, int] = {}
    for n in _own_nodes(func):
        if isinstance(n, ast.Return) and n.value is not None:
            for v in (n.value.elts if isinstance(n.value, ast.Tuple) else [n.value]):
                keys = _str_keys(v) if isinstance(v, ast.Dict) else local.get(v.id, set()) if isinstance(v, ast.Name) else set()
                for k in keys:
                    found.setdefault(k, n.lineno)
    return found


def _scan(project_root: pathlib.Path) -> dict[str, list[str]]:
    declared = _declared_keys(project_root / "pipeline" / "state.py")
    files = sorted((project_root / "nodes").glob("*.py")) + [project_root / "pipeline" / "graph.py"]
    offenders: dict[str, list[str]] = {}
    for path in files:
        if not path.exists():
            continue
        for f in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)):
            if not (f.name.endswith("_node") or f.name in HELPERS_RETURNING_STATE_UPDATES):
                continue
            for key, line in _returned_keys(f).items():
                if key not in declared:
                    offenders.setdefault(key, []).append(f"{path.name}:{line} ({f.name})")
    return offenders


@pytest.fixture(scope="module")
def offenders(project_root):
    return _scan(project_root)


def test_no_new_undeclared_state_keys(offenders):
    new = {k: v for k, v in offenders.items() if k not in KNOWN_UNDECLARED}
    assert not new, (
        "Nodes return keys that PipelineState doesn't declare — LangGraph silently drops them:\n"
        + "\n".join(f"  {k!r}  <- {', '.join(v)}" for k, v in sorted(new.items()))
        + "\nDeclare each on PipelineState (pipeline/state.py), or add it to KNOWN_UNDECLARED with a reason.")


def test_known_undeclared_list_has_no_stale_entries(offenders):
    stale = sorted(set(KNOWN_UNDECLARED) - set(offenders))
    assert not stale, f"Fixed already? Remove from KNOWN_UNDECLARED: {stale}"


def test_search_keys_are_declared(project_root):
    declared = _declared_keys(project_root / "pipeline" / "state.py")
    assert {"use_search", "search_notes"} <= declared


def test_scanner_itself_detects_an_undeclared_key(tmp_path):
    """Guard against the scanner rotting into a test that can never fail."""
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "nodes").mkdir()
    (tmp_path / "pipeline" / "state.py").write_text("class PipelineState:\n    known: int\n")
    (tmp_path / "nodes" / "n.py").write_text(
        "def some_node(state):\n    result = {'known': 1}\n    result['sneaky'] = 2\n    return result\n")
    assert list(_scan(tmp_path)) == ["sneaky"]


def test_scanner_ignores_returns_of_nested_helpers(tmp_path):
    """describe_node builds a request-body dict in a nested function; that is not a state update."""
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "nodes").mkdir()
    (tmp_path / "pipeline" / "state.py").write_text("class PipelineState:\n    known: int\n")
    (tmp_path / "nodes" / "n.py").write_text(
        "def some_node(state):\n"
        "    def _helper():\n"
        "        return {'logit_bias': 1, 'thinking': 2}\n"
        "    f = lambda: {'also_not_state': 1}\n"
        "    return {'known': _helper(), 'real_offender': 3}\n")
    assert list(_scan(tmp_path)) == ["real_offender"]