"""
pipeline/guards.py — deterministic checks that don't involve model calls.

Guards return (passed: bool, reason: str).
They run between pipeline stages to catch failures before downstream models see bad input.
"""

from __future__ import annotations

import logging
import re
from pipeline.state import PipelineState
from schemas.execution import DraftOutput, FixedOutput

log = logging.getLogger(__name__)

# ── Lazy evaluation guard ─────────────────────────────────────────────────────

# Patterns that indicate the 35B declared completion without generating code
_LAZY_PHRASES = [
    r"the\s+(code|implementation|function|class)\s+(is\s+)?already",
    r"already\s+(implemented|exists|present|complete|done)",
    r"no\s+changes?\s+(are\s+)?(needed|required|necessary)",
    r"this\s+(has\s+been|was\s+already)\s+(implemented|done|completed)",
    r"the\s+task\s+is\s+already\s+complete",
]
_LAZY_RE = re.compile("|".join(_LAZY_PHRASES), re.IGNORECASE)

# Minimum indicators that real code was produced
_CODE_INDICATORS = [
    r"def\s+\w+\s*\(",            # Python function
    r"class\s+\w+",               # Python class
    r"async\s+def\s+\w+",        # async function
    r"import\s+\w+",              # import statement
    r"from\s+\w+\s+import",       # from import
    r"```\w*\n",                  # code fence
    r"\w+\s*=\s*\w+\(",           # assignment with function call
]
_CODE_RE = re.compile("|".join(_CODE_INDICATORS))


def check_lazy_evaluation(draft: DraftOutput) -> tuple[bool, str]:
    """
    Check whether the 35B actually produced code rather than declaring
    the task complete without generating output.

    Returns (passed, reason).
    passed=True  → output looks genuine, proceed.
    passed=False → lazy evaluation detected, loop back to 35B.
    """
    if not draft.component_drafts:
        return False, "DraftOutput contains no component drafts"

    all_code = "\n".join(cd.code for cd in draft.component_drafts)

    if not all_code.strip():
        return False, "All component_drafts have empty code fields"

    # Check for lazy phrases in overall code
    lazy_match = _LAZY_RE.search(all_code)
    if lazy_match:
        return False, f"Lazy evaluation phrase detected: '{lazy_match.group(0)[:60]}'"

    # Check that at least some code indicators are present
    if not _CODE_RE.search(all_code):
        return False, "No code indicators found — output appears to be prose only"

    # Check notes field for explicit laziness admission
    if draft.implementation_notes:
        lazy_match = _LAZY_RE.search(draft.implementation_notes)
        if lazy_match:
            return False, f"Lazy phrase in implementation_notes: '{lazy_match.group(0)[:60]}'"

    return True, "ok"


def check_fixed_output_present(fixed: FixedOutput) -> tuple[bool, str]:
    """Verify Coder 14B produced actual fixed code, not empty output."""
    if not fixed.component_drafts:
        return False, "FixedOutput contains no component_drafts"

    all_code = "\n".join(cd.code for cd in fixed.component_drafts)
    if not all_code.strip():
        return False, "FixedOutput component_drafts all have empty code"

    return True, "ok"


# ── Iteration limit guard ──────────────────────────────────────────────────────

def check_iteration_limit(state: PipelineState, max_iterations: int) -> tuple[bool, str]:
    """
    Return (within_limit, reason).
    within_limit=False triggers unresolvable routing.
    """
    iteration = state.get("iteration", 0)
    if iteration >= max_iterations:
        return False, f"Iteration limit reached: {iteration}/{max_iterations}"
    return True, "ok"


# ── Sub-spec interface guard ───────────────────────────────────────────────────

def check_interface_compatibility(
    interfaces: list,
) -> tuple[bool, list[str]]:
    """
    Deterministic check that all sub-spec interface contracts are satisfied.

    Checks:
      1. Every shared object reference appears as an output in some sub-spec.
      2. Every consumer's input type has a matching provider output.

    Returns (all_satisfied, list_of_violation_messages).
    """
    violations: list[str] = []

    # Collect all declared outputs across all sub-specs
    all_outputs: set[str] = set()
    for iface in interfaces:
        all_outputs.update(iface.outputs)

    # Check shared object references
    for iface in interfaces:
        for ref in iface.shared_object_refs:
            if ref.name not in all_outputs:
                violations.append(
                    f"Sub-spec '{iface.component_name}' references shared object "
                    f"'{ref.name}' which is not declared as output by any sub-spec"
                )

    # Check input/output compatibility (type-level string match)
    all_output_types: set[str] = set()
    for iface in interfaces:
        all_output_types.update(iface.outputs)

    for iface in interfaces:
        for inp in iface.inputs:
            if inp not in all_output_types:
                violations.append(
                    f"Sub-spec '{iface.component_name}' expects input '{inp}' "
                    f"which is not provided by any sub-spec output"
                )

    return len(violations) == 0, violations

# ── Syntax validation guard ─────────────────────────────────────────

import ast
import traceback

def check_ast_syntax(code: str) -> tuple[bool, str]:
    """
    Deterministic check to ensure the generated Python code is syntactically valid.
    Uses Python's native AST parser.
    
    Returns (is_valid, error_traceback).
    """
    if not code.strip():
        return False, "Code is empty."
        
    try:
        ast.parse(code)
        return True, "ok"
    except SyntaxError as e:
        # Capture the exact line and error for the LLM to fix
        error_msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        return False, f"SyntaxError detected:\n{error_msg}"