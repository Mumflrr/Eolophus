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
  python run.py "/no-ensemble fix the null check in process()"

Prefix rules:
  /long             → force long mode
  /short            → force short mode
  /no-ensemble      → skip critique ensemble (any mode)
  /<mode>/<type>    → force both mode and task_type
  CLI flags take precedence over inline prefix if both are provided.

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
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



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

def parse_task_prefix(task: str) -> tuple[str, dict]:
    """
    Parse optional mode/type/no-ensemble prefix from the task string.

    Supported formats (case-insensitive, single leading slash):
      /long                   → {"mode": "long"}
      /short                  → {"mode": "short"}
      /no-ensemble            → {"no_ensemble": True}
      /long/coding            → {"mode": "long",  "task_type": "coding"}
      /long/ideation          → {"mode": "long",  "task_type": "ideation"}
      /long/mixed             → {"mode": "long",  "task_type": "mixed"}
      /short/coding           → {"mode": "short", "task_type": "coding"}
      /no-ensemble/long       → {"no_ensemble": True, "mode": "long"}

    Returns:
        (clean_task, overrides_dict)
    """
    import re

    MODES      = {"long", "short"}
    TASK_TYPES = {"coding", "ideation", "mixed"}

    FIRST_TOKEN  = r"(?:long|short|no-ensemble)"
    SECOND_TOKEN = r"(?:long|short|no-ensemble|coding|ideation|mixed)"
    pattern = re.compile(
        rf"^/({FIRST_TOKEN})(?:/({SECOND_TOKEN}))?\s+",
        re.IGNORECASE,
    )

    match = pattern.match(task)
    if not match:
        return task, {}

    tokens    = [t.lower() for t in (match.group(1), match.group(2)) if t]
    overrides: dict = {}

    for token in tokens:
        if token in MODES:
            overrides["mode"] = token
        elif token in TASK_TYPES:
            overrides["task_type"] = token
        elif token == "no-ensemble":
            overrides["no_ensemble"] = True

    clean_task = task[match.end():]
    return clean_task, overrides


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
    run_dir:   str,
    run_uuid:  str,
    args,
    task:      str,
    overrides: dict,
) -> None:
    import time
    run_meta = {
        "run_uuid":         run_uuid,
        "mode":             overrides.get("mode") or args.mode or "auto",
        "task_type":        overrides.get("task_type") or args.task_type or "auto",
        "no_ensemble":      overrides.get("no_ensemble", False) or args.no_ensemble,
        "image":            args.image,
        "started_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status":           "running",
        "task_preview":     task[:200],
        "prefix_overrides": overrides,
    }
    Path(run_dir, "run.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )


def build_initial_state(
    run_uuid:  str,
    run_dir:   str,
    task:      str,
    args,
    overrides: dict,
) -> dict:
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

    # Prefix overrides first, CLI flags win if set
    if overrides.get("mode"):
        state["mode"] = overrides["mode"]
    if overrides.get("task_type"):
        state["task_type"] = overrides["task_type"]
    if args.mode:
        state["mode"] = args.mode
    if args.task_type:
        state["task_type"] = args.task_type

    return state


def apply_flag_overrides(args, overrides: dict) -> None:
    no_ensemble = args.no_ensemble or overrides.get("no_ensemble", False)
    if no_ensemble:
        os.environ["PIPELINE_NO_ENSEMBLE"] = "1"

    effective_mode = args.mode or overrides.get("mode")
    if effective_mode == "short":
        os.environ["PIPELINE_FORCE_SHORT"] = "1"


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
        help    = "Force pipeline mode (default: auto-classified by 9B)",
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
    task, prefix_overrides = parse_task_prefix(task)

    if not task:
        print("ERROR: Task is empty after stripping prefix.", file=sys.stderr)
        return 1

    if prefix_overrides:
        log.info("Prefix overrides: %s", prefix_overrides)

    # ── 5. Initialise ─────────────────────────────────────────────────────────
    from storage.db import initialise as db_init
    db_init()

    apply_flag_overrides(args, prefix_overrides)

    # Now that we have the final task and args, write the run json
    write_run_json(run_dir, run_uuid, args, task, prefix_overrides)

    log.info("Run %s started", run_uuid)
    log.info("Run directory: %s", run_dir)
    log.info("Task: %s", task[:120] + ("..." if len(task) > 120 else ""))

    # ── 6. Build initial state ────────────────────────────────────────────────
    initial_state = build_initial_state(run_uuid, run_dir, task, args, prefix_overrides)

    # ── 7. Run pipeline ───────────────────────────────────────────────────────
    from pipeline.graph import get_graph
    app = get_graph()

    failed = False
    try:
        config      = {"run_name": run_uuid}
        final_state = app.invoke(initial_state, config=config)

        failed   = final_state.get("pipeline_failed", False)
        complete = final_state.get("pipeline_complete", False)

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