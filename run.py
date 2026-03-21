#!/usr/bin/env python3
"""
run.py — CLI entry point for the LLM orchestration pipeline.

Usage:
  python run.py "implement a FastAPI endpoint that..."
  python run.py --mode long --task-type coding "design a caching layer..."
  python run.py --image ./mockup.png "implement this UI"
  python run.py --no-ensemble --mode short "fix the typo in greet()"
  cat task.txt | python run.py --stdin

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
import uuid
from pathlib import Path


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
        handlers= [logging.StreamHandler(sys.stderr)],
    )


def make_run_dir(run_uuid: str) -> str:
    """Create and return the run directory path."""
    runs_root = Path(__file__).parent / "runs"
    runs_root.mkdir(exist_ok=True)
    run_dir   = runs_root / run_uuid
    run_dir.mkdir()
    return str(run_dir)


def write_run_json(run_dir: str, run_uuid: str, args, task: str) -> None:
    """Write initial run.json metadata."""
    import time
    run_meta = {
        "run_uuid":    run_uuid,
        "mode":        args.mode or "auto",
        "task_type":   args.task_type or "auto",
        "no_ensemble": args.no_ensemble,
        "image":       args.image,
        "started_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status":      "running",
        "task_preview":task[:200],
    }
    Path(run_dir, "run.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )


def build_initial_state(run_uuid: str, run_dir: str, task: str, args) -> dict:
    """Construct the initial PipelineState for this run."""
    state: dict = {
        "run_uuid":        run_uuid,
        "run_dir":         run_dir,
        "iteration":       0,
        "is_sub_spec":     False,
        "decompose":       False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
        "raw_text_input":  task,
        "normalised_input":task,
    }

    if args.image:
        image_path = str(Path(args.image).resolve())
        if not Path(image_path).exists():
            print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        state["raw_image_path"] = image_path

    # Override classification if flags provided
    if args.mode:
        state["mode"] = args.mode
    if args.task_type:
        state["task_type"] = args.task_type

    return state


def apply_flag_overrides(args) -> None:
    """Patch routing.yaml overrides via environment for this run."""
    if args.no_ensemble:
        os.environ["PIPELINE_NO_ENSEMBLE"] = "1"
    if args.mode == "short":
        os.environ["PIPELINE_FORCE_SHORT"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local LLM orchestration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs  = "?",
        help   = "Task description. If omitted, reads from --stdin or interactive prompt.",
    )
    parser.add_argument(
        "--mode",
        choices = ["short", "long"],
        default = None,
        help    = "Force pipeline mode (default: auto-classified by 9B)",
    )
    parser.add_argument(
        "--task-type",
        choices = ["coding", "ideation", "mixed"],
        default = None,
        dest    = "task_type",
        help    = "Force task type (default: auto-classified)",
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

    configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    # ── Get task text ─────────────────────────────────────────────────────────
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

    # ── Initialise ────────────────────────────────────────────────────────────
    from storage.db import initialise as db_init
    db_init()

    apply_flag_overrides(args)

    run_uuid = str(uuid.uuid4())
    run_dir  = make_run_dir(run_uuid)
    write_run_json(run_dir, run_uuid, args, task)

    log.info("Run %s started", run_uuid)
    log.info("Run directory: %s", run_dir)
    log.info("Task: %s", task[:120] + ("..." if len(task) > 120 else ""))

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state = build_initial_state(run_uuid, run_dir, task, args)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    from pipeline.graph import get_graph
    app = get_graph()

    try:
        config = {"run_name": run_uuid}
        final_state = app.invoke(initial_state, config=config)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user")
        _update_run_status(run_dir, "interrupted")
        return 130
    except Exception as e:
        log.error("Pipeline failed with unhandled exception: %s", e, exc_info=True)
        _update_run_status(run_dir, "error")
        return 1

    # ── Handle result ─────────────────────────────────────────────────────────
    failed   = final_state.get("pipeline_failed", False)
    complete = final_state.get("pipeline_complete", False)

    if failed:
        log.warning("Pipeline completed with UNRESOLVABLE status")
        reason = final_state.get("failure_reason", "unknown")
        print(f"\n[PIPELINE FAILED] {reason}", file=sys.stderr)
        _update_run_status(run_dir, "unresolvable")
    else:
        log.info("Pipeline completed successfully")
        _update_run_status(run_dir, "complete")

    # ── Output result ─────────────────────────────────────────────────────────
    output = _extract_output(final_state)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    print(f"\nRun ID: {run_uuid}", file=sys.stderr)
    print(f"Run dir: {run_dir}", file=sys.stderr)

    return 1 if failed else 0


def _extract_output(state: dict) -> str:
    """Extract the final output text from the pipeline state."""
    # Try fixed_output first, then draft_output as fallback
    fixed = state.get("fixed_output")
    if fixed and fixed.component_drafts:
        parts = []
        for cd in fixed.component_drafts:
            parts.append(f"# {cd.component_name}")
            parts.append(cd.code)
            parts.append("")
        return "\n".join(parts)

    draft = state.get("draft_output")
    if draft and draft.component_drafts:
        parts = []
        for cd in draft.component_drafts:
            parts.append(f"# {cd.component_name}")
            parts.append(cd.code)
            parts.append("")
        return "\n".join(parts)

    # Check for final output path
    final_path = state.get("final_output_path")
    if final_path and Path(final_path).exists():
        return Path(final_path).read_text(encoding="utf-8")

    return "[No output produced]"


def _update_run_status(run_dir: str, status: str) -> None:
    """Update the status field in run.json."""
    import time
    run_json_path = Path(run_dir) / "run.json"
    try:
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        data["status"]       = status
        data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
