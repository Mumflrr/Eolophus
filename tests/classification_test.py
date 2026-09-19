#!/usr/bin/env python3
"""
test_classify_cases.py — measure classify.yaml against the REAL classifier model.

Run from the Eolophus project root, with the classify model available
(9B server up, or model_manager able to start it):

    python test_classify_cases.py          # one pass
    python test_classify_cases.py -n 3     # 3 passes per case — the 9B samples at
                                           #   temperature > 0, so one pass can flatter a prompt
    python test_classify_cases.py -v       # print the model's own reasoning for misses

Only task_type is asserted: it is what routes (describe bypasses plan/draft/bugfix
entirely). The cases concentrate on the boundary that matters for search — a
request that USES search can be describe (the answer is information), coding
(the deliverable is software), ideation, or mixed. Where the boundary is
genuinely fuzzy, more than one type is accepted and the case says why.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import OrderedDict

# (group, task, accepted task_types, note)
CASES = [
    # ── search, answer wanted → describe ──────────────────────────────────
    ("search -> describe", "search for today's date", {"describe"}, ""),
    ("search -> describe", "use searxng to find today's date", {"describe"}, "tool named; still just a lookup"),
    ("search -> describe", "what's the latest stable release of FastAPI?", {"describe"}, ""),
    ("search -> describe", "can you search the internet?", {"describe"}, "capability question"),
    ("search -> describe", "compare the current top three Python web frameworks and recommend one", {"describe"}, "research, but the deliverable is a recommendation"),
    ("search -> describe", "how does SearXNG decide which search engines to query?", {"describe"}, "about search, wants an explanation"),

    # ── search involved, software wanted → coding ─────────────────────────
    ("search -> coding", "write a function that queries the SearXNG JSON API and returns the top 3 result titles", {"coding"}, "software ABOUT search"),
    ("search -> coding", "write a FastAPI app using the currently recommended startup and shutdown handling", {"coding"}, "search would only inform the build"),
    ("search -> coding", "look up the latest stable pydantic version and write a pyproject.toml that pins it", {"coding"}, "lookup feeds the build"),
    ("search -> coding", "add retry with exponential backoff to search_web in clients/search.py", {"coding"}, "existing search code"),

    # ── search involved, options wanted → ideation / mixed ────────────────
    ("search -> ideation/mixed", "explore approaches for adding web search to a local LLM pipeline; check what options exist today", {"ideation", "mixed"}, "options, no build verb"),
    ("search -> ideation/mixed", "explore ways to add response caching to this API, pick the best one and implement it", {"mixed", "coding"}, "fuzzy: mixed preferred, coding tolerable"),

    # ── no search: existing behaviour must not regress ────────────────────
    ("unchanged", "write a python function to reverse a string", {"coding"}, ""),
    ("unchanged", "what is the time complexity of quicksort?", {"describe"}, ""),
    ("unchanged", "explain how Python's GIL works", {"describe"}, ""),
    ("unchanged", "explore approaches for a distributed task queue", {"ideation"}, ""),
    ("unchanged", "describe the most notable features of this FEN: rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2", {"describe"}, ""),
]


def _val(x) -> str:
    """Enum or plain string -> plain string (TaskClassification uses use_enum_values)."""
    return "None" if x is None else str(getattr(x, "value", x))


def real_classifier(task: str, run_dir: str):
    from clients.llm import call_role
    from schemas.task_classification import TaskClassification
    return call_role(
        role="classify",
        template_vars={"task": task},
        response_schema=TaskClassification,
        stage="classify",
        run_dir=run_dir,
        thinking=False,
        max_retries=1,
        allow_escalation=False,   # measure the prompt on the classify model itself
    )


def run(cases, repeat: int, verbose: bool, classify_fn) -> int:
    run_dir = tempfile.mkdtemp(prefix="classify_cases_")
    groups: "OrderedDict[str, list]" = OrderedDict()
    failures = []

    for group, task, accepted, note in cases:
        hits, seen = 0, []
        for _ in range(repeat):
            try:
                c = classify_fn(task, run_dir)
                got, conf = _val(c.task_type), _val(c.confidence)
                seen.append((got, conf, getattr(c, "reasoning", "")))
                hits += got in accepted
            except Exception as e:                       # server down, schema failure, ...
                seen.append((f"ERROR: {type(e).__name__}", "-", str(e)[:200]))
        groups.setdefault(group, []).append((task, accepted, hits, seen, note))
        if hits < repeat:
            failures.append((task, accepted, seen))

    total = sum(r[2] for rows in groups.values() for r in rows)
    possible = sum(repeat for rows in groups.values() for _ in rows)

    for group, rows in groups.items():
        print(f"\n{group}")
        for task, accepted, hits, seen, note in rows:
            mark = "PASS" if hits == repeat else ("part" if hits else "FAIL")
            got = ",".join(sorted({s[0] for s in seen}))
            want = "/".join(sorted(accepted))
            short = task if len(task) <= 70 else task[:67] + "..."
            print(f"  [{mark}] {hits}/{repeat}  want {want:14s} got {got:22s} {short}")
            if note and hits < repeat:
                print(f"           note: {note}")

    print(f"\n{total}/{possible} passed ({100 * total // max(possible, 1)}%)")
    if failures and verbose:
        print("\n--- misses: what the model said and why ---")
        for task, accepted, seen in failures:
            print(f"\n* {task}")
            for got, conf, why in seen:
                print(f"    -> {got} (confidence={conf}): {why}")
    elif failures:
        print("(re-run with -v to see the model's reasoning for each miss)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--repeat", type=int, default=1, help="passes per case (default 1)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the model's reasoning for misses")
    args = ap.parse_args()
    return run(CASES, args.repeat, args.verbose, real_classifier)


if __name__ == "__main__":
    sys.exit(main())