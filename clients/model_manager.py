"""
clients/model_manager.py — on-demand model server lifecycle management.

For single-GPU setups where only one model fits in VRAM at a time.

Key design decisions:
  - Check if port is already alive BEFORE spawning anything.
    If the server is already running, just adopt it without spawning a new process.
  - Save pgid immediately at spawn time before the bash process can exit
    and its PID be recycled to something else.
  - Kill by port (lsof) rather than by pgid for adopted servers.
  - start_new_session=True isolates spawned servers from Python's session.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_config: dict = {}

def _cfg() -> dict:
    if not _config:
        p = Path(__file__).parent.parent / "config" / "models.yaml"
        with open(p) as f:
            _config.update(yaml.safe_load(f))
    return _config


# ── State ─────────────────────────────────────────────────────────────────────

_current_model:  Optional[str]              = None
_current_port:   Optional[int]              = None
_server_process: Optional[subprocess.Popen] = None
_server_pgid:    Optional[int]              = None   # saved at spawn, never re-fetched
_adopted:        bool                       = False  # True if we found a pre-existing server


# ── Models that cannot coexist on 10 GB VRAM ─────────────────────────────────

EXCLUSIVE_MODELS = {"9b", "27b", "35b", "deepcoder", "coder14b"}


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_model_loaded(model_id: str) -> None:
    """
    Ensure the specified model server is running and ready.
    Stops the current model first if a different one is needed.
    """
    global _current_model

    if _current_model == model_id:
        log.debug("Model %s already loaded", model_id)
        return

    if _current_model in EXCLUSIVE_MODELS and model_id in EXCLUSIVE_MODELS:
        log.info("Swapping model: %s -> %s", _current_model, model_id)
        _stop_current()
        time.sleep(3)
    else:
        log.info("Loading model: %s", model_id)

    _load_model(model_id)


def stop_all() -> None:
    """Stop all running model servers. Called at pipeline shutdown."""
    _stop_current()


def current_model() -> Optional[str]:
    return _current_model


# ── Internal ──────────────────────────────────────────────────────────────────

def _is_port_alive(port: int) -> bool:
    """Return True if a server is already responding on this port."""
    try:
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _load_model(model_id: str) -> None:
    """
    Load a model server. Adopts an existing running server if the port
    is already alive — avoids spawning a redundant process whose PID
    could be recycled before we stop it.
    """
    global _current_model, _current_port, _server_process, _server_pgid, _adopted

    cfg       = _cfg()
    model_cfg = cfg["models"][model_id]
    port      = model_cfg["port"]

    # ── Case 1: server already running on this port ───────────────────────────
    if _is_port_alive(port):
        log.info(
            "%s already running on port %d — adopting (no new process spawned)",
            model_cfg["name"], port,
        )
        _current_model  = model_id
        _current_port   = port
        _server_process = None
        _server_pgid    = None
        _adopted        = True
        return

    # ── Case 2: need to start a new server ───────────────────────────────────
    scripts = Path(__file__).parent.parent / "config" / "llama_flags"
    script  = scripts / f"{model_id}.sh"

    if not script.exists():
        raise FileNotFoundError(
            f"No launch script for model '{model_id}' at {script}"
        )

    log_dir  = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{model_id}_server.log"

    log.info("Starting %s on port %d...", model_cfg["name"], port)

    env = os.environ.copy()

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout            = lf,
            stderr            = lf,
            env               = env,
            start_new_session = True,   # own process group — no signal leakage
        )

    # Save pgid IMMEDIATELY before the bash process can exit and PID be recycled
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
        log.warning("Could not get pgid for %s — process exited immediately", model_id)

    _server_process = proc
    _server_pgid    = pgid
    _current_port   = port
    _adopted        = False

    _wait_for_health(port, model_cfg["name"], proc)
    _current_model = model_id
    log.info("%s ready on port %d", model_cfg["name"], port)


def _stop_current() -> None:
    """
    Stop the currently loaded model server.
    Uses saved pgid for managed processes, port-kill for adopted ones.
    """
    global _current_model, _current_port, _server_process, _server_pgid, _adopted

    if _current_model is None:
        return

    model_name = _current_model
    port       = _current_port

    log.info("Stopping %s (port %s)...", model_name, port)

    if _adopted or _server_pgid is None:
        # Server was pre-existing — just kill by port
        if port:
            _kill_port(port)
    else:
        # Server was spawned by us — kill its process group using saved pgid
        if _server_pgid is not None:
            try:
                os.killpg(_server_pgid, signal.SIGTERM)
                if _server_process:
                    _server_process.wait(timeout=20)
                log.debug("%s stopped via pgid %d", model_name, _server_pgid)
            except ProcessLookupError:
                log.debug("%s already gone (pgid %d)", model_name, _server_pgid)
            except subprocess.TimeoutExpired:
                log.warning("%s did not stop in 20s, sending SIGKILL", model_name)
                try:
                    os.killpg(_server_pgid, signal.SIGKILL)
                    if _server_process:
                        _server_process.wait()
                except Exception:
                    pass
            except Exception as e:
                log.warning("Error stopping %s: %s", model_name, e)

        # Belt-and-suspenders: also kill by port
        if port:
            _kill_port(port)

    os.system("pkill -9 -f llama-server")

    _server_process = None
    _server_pgid    = None
    _current_model  = None
    _current_port   = None
    _adopted        = False


def _kill_port(port: int) -> None:
    """Send SIGTERM to whatever process is listening on this port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        pids = [p.strip() for p in result.stdout.strip().split() if p.strip()]
        for pid_str in pids:
            try:
                pid = int(pid_str)
                # Safety: never kill our own Python process
                if pid == os.getpid():
                    log.warning("_kill_port: skipping own PID %d", pid)
                    continue
                os.kill(pid, signal.SIGTERM)
                log.debug("Sent SIGTERM to PID %d on port %d", pid, port)
            except (ProcessLookupError, ValueError):
                pass
            except Exception as e:
                log.warning("_kill_port error for PID %s: %s", pid_str, e)
    except Exception as e:
        log.warning("_kill_port lsof error: %s", e)


def _wait_for_health(
    port:    int,
    name:    str,
    proc:    subprocess.Popen,
    timeout: int = 300,
) -> None:
    """Poll health endpoint until the server responds or times out."""
    url      = f"http://localhost:{port}/health"
    elapsed  = 0
    interval = 3

    log.info("Waiting for %s to load (timeout: %ds)...", name, timeout)

    while elapsed < timeout:
        # Check if the process died before the server came up
        if proc.poll() is not None:
            raise RuntimeError(
                f"{name} process exited (code {proc.returncode}) before "
                f"becoming healthy. Check logs/{name.lower().replace(' ', '_')}_server.log"
            )

        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    log.info("%s loaded in %ds", name, elapsed)
                    return
        except Exception:
            pass

        time.sleep(interval)
        elapsed += interval

        if elapsed % 30 == 0:
            log.info("Still waiting for %s... (%ds elapsed)", name, elapsed)

    raise TimeoutError(
        f"{name} did not become healthy within {timeout}s. "
        f"Check logs/ for errors."
    )