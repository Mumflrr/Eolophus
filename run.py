#!/usr/bin/env python3
"""
run.py — CLI entry point for the LLM orchestration pipeline.

Usage:
  python run.py "implement a FastAPI endpoint that..."
  python run.py --mode long --task-type coding "design a caching layer..."
  python run.py --image ./mockup.png "implement this UI"
  python run.py --no-ensemble --mode short "fix the typo in greet()"
  cat task.txt | python run.py --stdin

Inline prefix syntax (parsed from the task string itself):
  python run.py "/long design a multi-tenant authentication system"
  python run.py "/short fix the typo in greet()"
  python run.py "/long/ideation explore approaches for a real-time collab tool"
  python run.py "/short/coding write a CSV parser with error handling"
  python run.py "/ultra/describe explain how the retry ladder works"
  python run.py "/describe/long explain the caching layer in depth"

Prefix rules — chain any number of tokens from up to three categories,
in any order, separated by '/', ending in whitespace before the task:
  mode:        long, short                         (informational only —
               see --mode's help below; does not select pipeline shape)
  profile:     auto, short, medium, long, ultra     (selects pipeline
               shape — this is what --mode used to be conflated with)
  task_type:   coding, ideation, mixed, describe

  Two tokens from the SAME category in one prefix (e.g. "/long/short") is
  an error, not last-wins — run.py will refuse to guess which you meant.
  An unrecognised token immediately after a leading '/' is left alone and
  treated as ordinary task text (so a task that happens to start with a
  slash for unrelated reasons isn't mangled) UNLESS it's a near-miss of a
  known token (e.g. "/lng"), in which case a warning is printed — the
  prefix is still left as plain text, but you'll know why nothing changed.

  CLI flags take precedence over inline prefix if both are provided.

Retired flags (recognised for backward compatibility, but currently
inert — see routers.py/runs.py's own comments for why): --no-ensemble,
/no-ensemble. Using either prints a loud warning rather than silently
appearing to work.

Environment variables:
  PIPELINE_DB       Path to SQLite database (default: ~/.pipeline/pipeline.db)
  LANGFUSE_HOST     Langfuse server URL (default: http://localhost:3000)
  MODEL_DIR         Directory containing GGUF model files
  LOG_LEVEL         Logging level: DEBUG/INFO/WARNING/ERROR (default: INFO)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from pipeline.graph import get_graph
from langgraph.types import Command

# ── Terminal tee ──────────────────────────────────────────────────────────────

class _TeeLogger:
    """
    Writes to both the original stream and a log file simultaneously.
    Captures all terminal output into runs/{uuid}/terminal.log
    without suppressing live output.
    """
    def __init__(self, stream, logfile):
        self._stream  = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)
        self._logfile.flush()

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def isatty(self):
        return hasattr(self._stream, 'isatty') and self._stream.isatty()


def _install_terminal_tee(run_dir: str):
    """Install tee on stdout+stderr → runs/{uuid}/terminal.log."""
    log_path     = os.path.join(run_dir, "terminal.log")
    terminal_log = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout   = _TeeLogger(sys.__stdout__, terminal_log)
    sys.stderr   = _TeeLogger(sys.__stderr__, terminal_log)
    return terminal_log


def _restore_terminal(terminal_log) -> None:
    """Restore stdout/stderr to original streams and close the log file."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        terminal_log.close()
    except Exception:
        pass


# ── Prefix parser ─────────────────────────────────────────────────────────────

class PrefixError(ValueError):
    """Raised for an unambiguous inline-prefix mistake — two tokens from
    the same category (e.g. "/long/short") — never for a merely-unknown
    token, which is treated as ordinary task text instead (see
    parse_task_prefix's docstring)."""


# Category -> recognised tokens. A token appears in exactly one category;
# MODES/PROFILES overlap on "long"/"short" by name only — which category
# a given occurrence belongs to is decided by _CATEGORY_OF below, and a
# prefix chain may carry at most one mode-category token and one
# profile-category token (they're independent axes: mode is the
# informational short/long field on TaskClassification; profile is what
# actually selects pipeline shape — see classifier.py's profile-resolution
# comment). DEPRECATED lists directives that still parse (so old scripts
# don't hard-fail) but do nothing — see this module's docstring.
_MODES      = {"long", "short"}
_PROFILES   = {"auto", "short", "medium", "long", "ultra"}
_TASK_TYPES = {"coding", "ideation", "mixed", "describe"}
_DEPRECATED = {"no-ensemble"}

# Longest-first so e.g. "no-ensemble" isn't cut short by a shorter
# accidental prefix match against some future token.
_ALL_TOKENS = sorted(_MODES | _PROFILES | _TASK_TYPES | _DEPRECATED, key=len, reverse=True)
_TOKEN_RE   = "|".join(re.escape(t) for t in _ALL_TOKENS)


def _levenshtein(a: str, b: str) -> int:
    """Small bespoke edit distance — used only for the "did you mean"
    warning below, not worth a dependency for one heuristic check."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _category_of(token: str) -> str:
    """
    Which category a recognised token belongs to. "long"/"short" are
    ambiguous by spelling alone (they're valid MODE and PROFILE tokens);
    resolved as MODE here because that's this syntax's original meaning
    and its more common use, and PROFILE has its own unambiguous-only
    tokens (auto/medium/ultra) for when the caller means profile instead.
    A caller who needs "/long" to mean profile rather than mode should
    use --profile long on the command line, which is unambiguous.
    """
    if token in _MODES:
        return "mode"
    if token in _TASK_TYPES:
        return "task_type"
    if token in _PROFILES:  # only auto/medium/ultra reach here — long/short claimed above
        return "profile"
    if token in _DEPRECATED:
        return "deprecated"
    raise AssertionError(f"unreachable — {token!r} matched _TOKEN_RE but no category")


def parse_task_prefix(task: str) -> tuple[str, dict, list[str]]:
    """
    Parse an optional chain of mode/profile/task_type/deprecated-flag
    tokens from the front of the task string.

    Supported formats (case-insensitive, chain any number of tokens from
    up to three categories, in any order, separated by '/'):
      /long                        → {"mode": "long"}
      /ultra                       → {"profile": "ultra"}
      /describe                    → {"task_type": "describe"}
      /long/describe               → {"mode": "long", "task_type": "describe"}
      /ultra/describe              → {"profile": "ultra", "task_type": "describe"}
      /no-ensemble/long            → {"mode": "long"}  (no-ensemble reported as deprecated, not an override)

    Two tokens from the SAME category (e.g. "/long/short", "/coding/mixed")
    raise PrefixError — this is unambiguous misuse, not last-wins.

    An unrecognised token right after the leading '/' is NOT an error: it's
    left as part of the task text (a task that genuinely starts with a
    slash for unrelated reasons — a file path, a command name — must not
    be mangled). If that unrecognised token is a close misspelling of a
    real one (edit distance <= 2, e.g. "/lng"), a warning string is
    returned (third element) so the caller can print it — the task text
    is still left untouched either way.

    Returns:
        (clean_task, overrides, warnings) — overrides has at most one of
        each "mode"/"profile"/"task_type" key; warnings is a list of
        human-readable strings (deprecated-flag notices, near-miss
        typo notices), empty when there's nothing to report.
    """
    warnings: list[str] = []

    # Only a leading '/' followed IMMEDIATELY (no space) by token chars is
    # even candidate syntax — this must stay conservative, since anything
    # here that isn't part of a genuine directive chain has to fall through
    # to plain task text unchanged.
    lead_match = re.match(r"^/([A-Za-z][A-Za-z0-9-]*(?:/[A-Za-z][A-Za-z0-9-]*)*)\s+", task)
    if not lead_match:
        return task, {}, warnings

    raw_chain  = lead_match.group(1)
    raw_tokens = raw_chain.split("/")

    # All-or-nothing: if EVERY token in the chain is recognised, it's a
    # directive prefix. If even one isn't, the whole thing is almost
    # certainly not a directive chain at all (a path, a hashtag-style
    # opener, etc.) — better to leave the full original text alone than
    # to consume the tokens that happened to match and mangle the rest.
    lowered = [t.lower() for t in raw_tokens]
    if not all(re.fullmatch(_TOKEN_RE, t, re.IGNORECASE) for t in lowered):
        # Near-miss check: only on a SINGLE-token chain immediately
        # followed by whitespace — a multi-token chain with one bad token
        # is far more likely coincidental (unrelated task text) than a
        # typo'd directive, and guessing there risks a false alarm on
        # ordinary text that happens to contain a slash.
        if len(lowered) == 1 and 2 <= len(lowered[0]) <= 14:
            candidate = lowered[0]
            near = [t for t in _ALL_TOKENS if _levenshtein(candidate, t) <= 2]
            if near:
                warnings.append(
                    f"'/{raw_tokens[0]}' isn't a recognised prefix token "
                    f"(did you mean '/{near[0]}'?) — left as plain task text."
                )
        return task, {}, warnings

    overrides: dict = {}
    seen_categories: dict[str, str] = {}
    for original, token in zip(raw_tokens, lowered):
        category = _category_of(token)
        if category == "deprecated":
            warnings.append(
                f"'/{original}' is recognised but retired — it currently has no "
                f"effect on the pipeline (see routers.py's ensemble-gating "
                f"comment). Remove it or open an issue if you need it back."
            )
            continue
        if category in seen_categories:
            raise PrefixError(
                f"Prefix has two {category} tokens: '/{seen_categories[category]}' "
                f"and '/{original}' — only one {category} directive is allowed "
                f"per task. Pick one."
            )
        seen_categories[category] = original
        overrides[category] = token

    clean_task = task[lead_match.end():]
    return clean_task, overrides, warnings


# ── Helpers ───────────────────────────────────────────────────────────────────

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level    = getattr(logging, level.upper(), logging.INFO),
        format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [logging.StreamHandler(sys.stderr)],
    )


def make_run_dir(run_uuid: str) -> tuple[str, object]:
    """
    Create run directory, install terminal tee.
    Returns (run_dir_path, terminal_log_handle).
    """
    runs_root = Path(__file__).parent / "runs"
    runs_root.mkdir(exist_ok=True)
    run_dir   = runs_root / run_uuid
    run_dir.mkdir()
    terminal_log = _install_terminal_tee(str(run_dir))
    return str(run_dir), terminal_log


def write_run_json(
    run_dir:    str,
    run_uuid:   str,
    args,
    task:       str,
    overrides:  dict,
    warnings:   list[str],
    raw_prefix: Optional[str],
) -> None:
    """
    Write the initial run.json. "Developer mode": records what was asked
    for (CLI flags, inline prefix, and which one WON when both were given
    for the same category) as its own explicit block, distinct from what
    classify_node eventually resolves — see _enrich_run_json_with_resolution
    below, which appends the resolved side after the pipeline finishes, so
    a completed run.json shows requested-vs-resolved side by side, the
    same distinction pipeline/state.py's requested_task_type exists for.
    """
    import time

    # CLI flags win over inline prefix per-category, independently — a
    # task can legitimately set --task-type on the command line while
    # using the inline prefix only for mode, and each category's winner
    # should be reported, not just an overall "CLI wins" flag.
    def resolved(category: str, cli_value):
        prefix_value = overrides.get(category)
        if cli_value:
            return cli_value, ("cli" if not prefix_value else "cli (overrode prefix)")
        if prefix_value:
            return prefix_value, "prefix"
        return None, None

    mode_val,     mode_src     = resolved("mode",      args.mode)
    profile_val,  profile_src  = resolved("profile",   args.profile)
    task_type_val, task_type_src = resolved("task_type", args.task_type)
    no_ensemble = bool(args.no_ensemble or "no-ensemble" in (overrides.get("_deprecated_used") or []))

    run_meta = {
        "run_uuid":    run_uuid,
        "requested": {
            "mode":       {"value": mode_val,      "source": mode_src},
            "profile":    {"value": profile_val,   "source": profile_src},
            "task_type":  {"value": task_type_val, "source": task_type_src},
        },
        # Flat legacy-shape fields kept alongside "requested" above for
        # anything reading the old flat keys directly — "auto" fills in
        # only here, matching the pre-existing on-disk contract.
        "mode":             mode_val or "auto",
        "task_type":        task_type_val or "auto",
        "profile":          profile_val or "auto",
        "no_ensemble":      no_ensemble,
        "no_ensemble_note": (
            "requested but currently has no effect — see routers.py's "
            "ensemble-gating comment" if no_ensemble else None
        ),
        "image":            args.image,
        "raw_prefix":       raw_prefix,
        "prefix_overrides": {k: v for k, v in overrides.items() if not k.startswith("_")},
        "parse_warnings":   warnings,
        "started_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status":           "running",
        "task_preview":     task[:200],
    }
    Path(run_dir, "run.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )


def _enrich_run_json_with_resolution(run_dir: str, final_state: dict) -> None:
    """
    Append what classify_node actually decided, alongside what was
    requested (written by write_run_json above) — the same "requested vs.
    resolved" comparison this whole run.py pass exists to make visible.
    Best-effort: run.json already has enough for the pipeline to have run
    correctly without this, so a failure here logs and moves on rather
    than turning a successful pipeline run into a reported failure.
    """
    run_json_path = Path(run_dir) / "run.json"
    try:
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        classification = final_state.get("classification")
        classification_dict = None
        if classification is not None:
            classification_dict = (
                classification.model_dump() if hasattr(classification, "model_dump")
                else dict(classification) if isinstance(classification, dict)
                else None
            )
        data["resolved"] = {
            "mode":            final_state.get("mode"),
            "task_type":       final_state.get("task_type"),
            "profile":         final_state.get("profile"),
            "classifier_confidence": final_state.get("classifier_confidence"),
            "classification":  classification_dict,
        }
        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to enrich run.json with resolved classification: %s", e
        )


def build_initial_state(
    run_uuid:  str,
    run_dir:   str,
    task:      str,
    args,
    overrides: dict,
) -> dict:
    """
    Pins go through requested_mode/requested_profile/requested_task_type,
    NOT the raw mode/task_type/profile keys — those three are
    classify_node's OWN output keys (see its final return dict), so
    writing a caller pin under the same name would be indistinguishable
    from classify_node's prior output the moment this state is ever
    re-invoked on the same thread. run.py currently never does that (each
    CLI invocation is a fresh thread_id, single classify pass — see
    main()'s Pause & Resume Loop, which resumes an interrupted node via
    Command(resume=...), not a fresh classify), so this collision is
    latent rather than active here today. Routed through the same
    requested_* fields as api/routers/runs.py's identical fix anyway:
    matching the one correct pattern everywhere it could apply beats
    leaving a known bug shape sitting inert until some future feature
    (e.g. a CLI "amend and re-run" mode) reuses a thread and revives it.
    """
    state: dict = {
        "run_uuid":          run_uuid,
        "run_dir":           run_dir,
        "iteration":         0,
        "is_sub_spec":       False,
        "decompose":         False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
        "raw_text_input":    task,
        "normalised_input":  task,
    }

    if args.image:
        image_path = str(Path(args.image).resolve())
        if not Path(image_path).exists():
            print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        state["raw_image_path"] = image_path

    # Prefix overrides first, CLI flags win if set — independently per
    # category, so e.g. --task-type alone doesn't also have to repeat
    # whatever mode the prefix already specified.
    requested_mode      = args.mode      or overrides.get("mode")
    requested_profile   = args.profile   or overrides.get("profile")
    requested_task_type = args.task_type or overrides.get("task_type")

    if requested_mode:
        state["requested_mode"] = requested_mode
    if requested_profile and requested_profile != "auto":
        state["requested_profile"] = requested_profile
    if requested_task_type:
        state["requested_task_type"] = requested_task_type

    return state


def apply_flag_overrides(args, warnings: list[str]) -> None:
    """
    PIPELINE_NO_ENSEMBLE / PIPELINE_FORCE_SHORT / PIPELINE_ULTRA are
    retired (see routers.py's ensemble-gating comment and runs.py's
    "no longer carries ..." comment) — nothing reads them anymore.
    Setting them here used to silently do nothing while looking like it
    worked. Now: --no-ensemble prints a loud warning instead of setting a
    dead env var, and --mode/prefix "ultra" is refused outright, because
    "ultra" was never a valid `mode` value to begin with (TaskClassification.mode
    is short|long only — ultra is a PROFILE, and --profile ultra /
    "/ultra" already exists and actually works, via requested_profile).
    """
    if args.no_ensemble:
        warnings.append(
            "--no-ensemble is recognised but retired — it currently has no "
            "effect on the pipeline (see routers.py's ensemble-gating "
            "comment). This run will use the normal ensemble rules."
        )

    if args.mode == "ultra":
        print(
            "ERROR: --mode ultra is not valid — mode is short|long only "
            "(TaskClassification.mode). Ultra is a PIPELINE PROFILE, not a "
            "mode: use --profile ultra instead.",
            file=sys.stderr,
        )
        sys.exit(2)


def _extract_output(state: dict) -> str:
    """Extract final output text from pipeline state."""

    # Describe tasks write a final.json with an "answer" key
    final_path = state.get("final_output_path")
    if final_path and Path(final_path).exists():
        raw = Path(final_path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            if "answer" in data:
                return data["answer"]
        except Exception:
            pass
        return raw

    # Coding tasks: prefer fixed_output, fall back to draft_output
    fixed = state.get("fixed_output")
    if fixed and getattr(fixed, "component_drafts", None):
        parts = []
        for cd in fixed.component_drafts:
            parts.append(f"# {cd.component_name}")
            parts.append(cd.code)
            parts.append("")
        return "\n".join(parts)

    draft = state.get("draft_output")
    if draft and getattr(draft, "component_drafts", None):
        parts = []
        for cd in draft.component_drafts:
            parts.append(f"# {cd.component_name}")
            parts.append(cd.code)
            parts.append("")
        return "\n".join(parts)

    return "[No output produced]"


def _update_run_status(run_dir: str, status: str) -> None:
    import time
    run_json_path = Path(run_dir) / "run.json"
    try:
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        data["status"]       = status
        data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description       = "Local LLM orchestration pipeline",
        formatter_class   = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs   = "?",
        help    = "Task description. Omit to read from --stdin or interactive prompt.",
    )
    parser.add_argument(
        "--mode",
        choices = ["short", "long"],
        default = None,
        help    = (
            "Pin the informational mode field (TaskClassification.mode). "
            "Does NOT select pipeline shape on its own — use --profile for "
            "that. 'ultra' is not a valid mode; use --profile ultra."
        ),
    )
    parser.add_argument(
        "--profile",
        choices = ["auto", "short", "medium", "long", "ultra"],
        default = None,
        help    = (
            "Pin the pipeline profile (node set / model tier / iteration "
            "budget — see config/routing.yaml's pipeline_profiles). This is "
            "what actually selects pipeline shape. 'ultra' = overnight "
            "deep-thinking via each stage's escalation-ladder final model "
            "(see config/models.yaml's escalation_ladders); requires those "
            "models to be downloaded and, for 27b_ultra specifically, an "
            "MTP-capable GGUF + llama.cpp built from PR #22673."
        ),
    )
    parser.add_argument(
        "--task-type",
        choices = ["coding", "ideation", "mixed", "describe"],
        default = None,
        dest    = "task_type",
        help    = "Force task type (default: auto-classified). Use 'describe' for analysis/explanation tasks.",
    )
    parser.add_argument(
        "--image",
        default = None,
        help    = "Path to input image for vision decode",
    )
    parser.add_argument(
        "--no-ensemble",
        action  = "store_true",
        default = False,
        dest    = "no_ensemble",
        help    = "Skip critique ensemble (faster, less thorough)",
    )
    parser.add_argument(
        "--stdin",
        action  = "store_true",
        default = False,
        help    = "Read task from stdin",
    )
    parser.add_argument(
        "--output",
        default = None,
        help    = "Write final output to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--log-level",
        default = os.environ.get("LOG_LEVEL", "INFO"),
        dest    = "log_level",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # ── 1. SETUP RUN DIR & TEE LOGGER ─────────────────────────────────────────
    # Do this before configuring logging so the logging module binds
    # to the _TeeLogger instead of the raw sys.stderr.
    import uuid
    run_uuid = str(uuid.uuid4())
    run_dir, terminal_log = make_run_dir(run_uuid)

    # ── 2. CONFIGURE LOGGING ──────────────────────────────────────────────────
    configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    # ── 3. Get task text ──────────────────────────────────────────────────────
    if args.stdin:
        task = sys.stdin.read().strip()
    elif args.task:
        task = args.task.strip()
    else:
        print("Enter task (Ctrl+D when done):", file=sys.stderr)
        task = sys.stdin.read().strip()

    if not task:
        print("ERROR: No task provided.", file=sys.stderr)
        return 1

    # ── 4. Parse inline prefix ────────────────────────────────────────────────
    raw_task_before_prefix = task
    try:
        task, prefix_overrides, prefix_warnings = parse_task_prefix(task)
    except PrefixError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not task:
        print("ERROR: Task is empty after stripping prefix.", file=sys.stderr)
        return 1

    raw_prefix = raw_task_before_prefix[:len(raw_task_before_prefix) - len(task)].strip() or None

    if prefix_overrides:
        log.info("Prefix overrides: %s", prefix_overrides)
    for w in prefix_warnings:
        log.warning(w)
        print(f"WARNING: {w}", file=sys.stderr)

    # ── 5. Initialise ─────────────────────────────────────────────────────────
    from storage.db import initialise as db_init
    db_init()

    flag_warnings: list[str] = []
    apply_flag_overrides(args, flag_warnings)
    for w in flag_warnings:
        log.warning(w)
        print(f"WARNING: {w}", file=sys.stderr)

    # Now that we have the final task and args, write the run json
    write_run_json(
        run_dir, run_uuid, args, task, prefix_overrides,
        prefix_warnings + flag_warnings, raw_prefix,
    )

    log.info("Run %s started", run_uuid)
    log.info("Run directory: %s", run_dir)
    log.info("Task: %s", task[:120] + ("..." if len(task) > 120 else ""))

    # ── 6. Build initial state ────────────────────────────────────────────────
    initial_state = build_initial_state(run_uuid, run_dir, task, args, prefix_overrides)

    # ── 7. Run pipeline ───────────────────────────────────────────────────────
    app, callbacks = get_graph()

    failed = False
    try:
        # NEW: Checkpointers require a 'thread_id' in the configurable block
        # to know exactly where to save and resume the memory state.
        config = {
            "configurable": {"thread_id": run_uuid}, 
            "run_name": run_uuid
        }
        if callbacks:
            config["callbacks"] = callbacks
        
        # Start the initial run
        final_state = app.invoke(initial_state, config=config)

        # NEW: The Pause & Resume Loop
        while True:
            # Check the current status of the graph
            state_snapshot = app.get_state(config)
            
            # If there are no pending tasks (next == []), the graph is completely finished
            if not state_snapshot.next:
                final_state = state_snapshot.values
                break

            # If it's paused, check if it was our gatekeeper's interrupt()
            pending_tasks = state_snapshot.tasks
            if pending_tasks and pending_tasks[0].interrupts:
                # Extract the question we yielded from gatekeeper.py
                question = pending_tasks[0].interrupts[0].value
                print(f"\n[PIPELINE PAUSED] {question}", file=sys.stderr)
                
                # Prompt you in the terminal
                user_response = input("> ")
                
                # Resume the graph, passing your answer directly back into gatekeeper.py
                final_state = app.invoke(Command(resume=user_response), config=config)
            else:
                # Graph stopped for some other reason (e.g., hit max recursion limit)
                final_state = state_snapshot.values
                break

        failed   = final_state.get("pipeline_failed", False)
        complete = final_state.get("pipeline_complete", False)

        # Developer-mode visibility: what classify_node actually decided,
        # next to what was requested (written by write_run_json earlier).
        # Best-effort — see this function's own docstring.
        _enrich_run_json_with_resolution(run_dir, final_state)

        if failed:
            log.warning("Pipeline completed with UNRESOLVABLE status")
            reason = final_state.get("failure_reason", "unknown")

            # Confidence clarification — surface the question clearly
            if reason and "?" in reason:
                print(f"\n[CLARIFICATION NEEDED] {reason}", file=sys.stderr)
            else:
                print(f"\n[PIPELINE FAILED] {reason}", file=sys.stderr)
            _update_run_status(run_dir, "unresolvable")
        else:
            log.info("Pipeline completed successfully")
            _update_run_status(run_dir, "complete")

        # ── Output result ──────────────────────────────────────────────────
        output = _extract_output(final_state)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)

    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user")
        _update_run_status(run_dir, "interrupted")
        failed = True
    except Exception as e:
        log.error("Pipeline failed with unhandled exception: %s", e, exc_info=True)
        _update_run_status(run_dir, "error")
        failed = True
    finally:
        print(f"\nRun ID: {run_uuid}", file=sys.stderr)
        print(f"Run dir: {run_dir}", file=sys.stderr)

        # Flush telemetry before exiting
        try:
            from langfuse import Langfuse
            langfuse_client = Langfuse()
            langfuse_client.flush()
            log.debug("Langfuse telemetry flushed successfully.")
        except ImportError:
            pass
        except Exception as e:
            log.warning("Failed to flush Langfuse telemetry: %s", e)

        # Stop model servers before restoring terminal so final log lines
        # are captured in terminal.log
        try:
            from clients.model_manager import stop_all
            stop_all()
        except Exception:
            pass

        # Restore stdout/stderr last — after all logging is done
        _restore_terminal(terminal_log)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())