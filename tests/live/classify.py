#!/usr/bin/env python3
"""
tests/live/classify.py — measure classify.yaml against the REAL classifier model.

NEEDS A LIVE MODEL. Not part of `pytest` (that suite is offline and lives in tests/).

Run from anywhere; the project root (the directory holding clients/ and schemas/)
is found automatically, so it works from any directory. The
classify model must be reachable (9B server up, or model_manager able to start it):

    python tests/live/classify.py          # one pass
    python tests/live/classify.py -n 3     # 3 passes per case — the 9B samples at
                                                 #   temperature > 0, so one pass can flatter a prompt
    python tests/live/classify.py -v       # print the model's own reasoning for misses
    python tests/live/classify.py --cap 800  # abort any call that generates more than 800 tokens

A classification is ~150 tokens of JSON, so --cap (default 2000, vs routing.yaml's 6000) makes a
runaway fail in seconds instead of grinding to the full cap. When a call hits it, the script shows
what the model produced (thinking vs answer) — with -v, an excerpt of both.

The offline checks of classify.yaml live in tests/test_classify_prompt.py; the logic of THIS script is
unit-tested (offline) in tests/test_tests/live_classify.py.

Only task_type is asserted: it is what routes (describe bypasses plan/draft/bugfix
entirely). The cases concentrate on the boundary that matters for search — a
request that USES search can be describe (the answer is information), coding
(the deliverable is software), ideation, or mixed. Where the boundary is
genuinely fuzzy, more than one type is accepted and the case says why.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
from collections import OrderedDict


def find_project_root(start: pathlib.Path) -> pathlib.Path:
    """
    Walk up from `start` to the first directory that contains both clients/ and schemas/.
    Running `python ./tests/x.py` puts tests/ on sys.path, NOT the project root — which is
    why `import clients` used to fail with ModuleNotFoundError.
    """
    for candidate in (start, *start.parents):
        if (candidate / "clients").is_dir() and (candidate / "schemas").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Couldn't find the project root (a directory containing clients/ and schemas/) above {start}")


try:
    PROJECT_ROOT = find_project_root(pathlib.Path(__file__).resolve().parent)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except FileNotFoundError as _e:          # reported clearly by main(); keeps `import` of this module safe
    PROJECT_ROOT = None
    _ROOT_ERROR = str(_e)

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


DEFAULT_CAP = 2000


def real_classifier(task: str, run_dir: str, cap: int = DEFAULT_CAP):
    from clients.llm import call_role
    from schemas.task_classification import TaskClassification
    return call_role(
        role="classify",
        template_vars={"task": task},
        response_schema=TaskClassification,
        stage="classify",
        run_dir=run_dir,
        thinking=False,
        max_retries=0,             # same as classify_node
        allow_escalation=False,    # measure the prompt on the classify model itself
        output_cap_override=cap,   # fail fast on a runaway (routing.yaml's classify cap is 6000)
    )


def _excerpt(text: str, head: int = 300, tail: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= head + tail + 5 else f"{text[:head]} [...] {text[-tail:]}"


def describe_error(e: Exception) -> tuple[str, str]:
    """(label, detail). A TruncatedOutputError says what the model spent its tokens on."""
    label, detail = f"ERROR: {type(e).__name__}", str(e)[:200]
    thinking, partial = getattr(e, "thinking_block", None), getattr(e, "partial_answer", None)
    if thinking is not None or partial is not None:            # duck-typed: clients.llm.TruncatedOutputError
        detail += (f"\n      before the cap it wrote {len(thinking or '')} chars of THINKING "
                   f"and {len(partial or '')} chars of ANSWER")
        if thinking:
            detail += f"\n      thinking: {_excerpt(thinking)}"
        if partial:
            detail += f"\n      answer:   {_excerpt(partial)}"
    return label, detail


class SystemicFailure(Exception):
    """Every attempt on the first case failed with the same error — setup problem, not a prompt problem."""


def run(cases, repeat: int, verbose: bool, classify_fn) -> int:
    run_dir = tempfile.mkdtemp(prefix="classify_cases_")
    groups: "OrderedDict[str, list]" = OrderedDict()
    failures = []

    for index, (group, task, accepted, note) in enumerate(cases):
        hits, seen = 0, []
        for _ in range(repeat):
            try:
                c = classify_fn(task, run_dir)
                got, conf = _val(c.task_type), _val(c.confidence)
                seen.append((got, conf, getattr(c, "reasoning", "")))
                hits += got in accepted
            except Exception as e:                       # server down, schema failure, truncation, ...
                label, detail = describe_error(e)
                seen.append((label, "-", detail))
        # If the very first case errored identically on every attempt, the other 16 would too:
        # stop and say what's wrong rather than printing a wall of identical failures.
        if (index == 0 and all(s[0].startswith("ERROR") for s in seen)
                and len({(s[0], s[2].splitlines()[0]) for s in seen}) == 1):
            raise SystemicFailure(f"{seen[0][0]}: {seen[0][2]}")
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
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"max tokens per classification call (default {DEFAULT_CAP}; routing.yaml uses 6000)")
    args = ap.parse_args()

    if PROJECT_ROOT is None:
        print(f"ERROR: {_ROOT_ERROR}", file=sys.stderr)
        return 2
    try:
        import clients.llm, schemas.task_classification  # noqa: F401  (fail fast, with the real reason)
    except Exception as e:
        print(f"ERROR: couldn't import the pipeline from {PROJECT_ROOT}: {type(e).__name__}: {e}\n"
              f"       (activate the same environment you run the server in)", file=sys.stderr)
        return 2

    try:
        return run(CASES, args.repeat, args.verbose, lambda task, run_dir: real_classifier(task, run_dir, args.cap))
    except SystemicFailure as e:
        print(f"\nAborted: the first case failed the same way on every attempt, so the rest would too:\n  {e}\n",
              file=sys.stderr)
        if "truncated" in str(e).lower():
            print("The model ran out of tokens on a ~150-token task. If the THINKING figure above is large, the classify\n"
                  "stage is thinking despite thinking=False (run tests/live/thinking_control.py to see which switch your\n"
                  "llama-server honours). If the ANSWER figure is large, the model is rambling in the JSON itself.",
                  file=sys.stderr)
        else:
            print("Check that the classify model server is up (e.g. curl http://localhost:8081/health) and that "
                  "you're in the environment the pipeline runs in.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())