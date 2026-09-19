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

import json
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

EXCLUSIVE_MODELS = {"9b", "27b", "27b_ultra", "35b", "deepcoder"}


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_model_loaded(model_id: str) -> bool:
    """
    Ensure the specified model server is running and ready.
    Stops the current model first if a different one is needed.

    Returns True if a real load just happened (fresh spawn, or a
    flash-swap from a different model) — False if model_id was already
    the currently-loaded model and this call was a no-op. Callers that
    want to capture load-time data (e.g. clients/model_memory.py reading
    the just-written server log for memory buffer sizes) should only do
    so when this returns True — re-reading the log after a no-op call
    would just re-report the same numbers from whenever the model
    actually loaded, misleadingly attributed to the current call.

    NOTE: an ADOPTED server (Case 1 in _load_model — a server already
    listening on the target port when we checked) returns True here even
    though this call didn't perform the load itself. That's intentional,
    not an oversight: we still don't know when that server's log-file
    load block happened, and log-parsing (clients/model_memory.py) reads
    "the last load block in the file" regardless — for an adopted server,
    the log file's last block IS still that server's actual startup, just
    from an earlier invocation of this process (or a different one
    entirely). It's the right data, just not freshly produced by THIS
    call. If you need to distinguish "genuinely fresh spawn" from
    "adopted", check the module-level _adopted flag separately — this
    return value answers "is there load data worth reading", not "did
    THIS call spawn the process".
    """
    global _current_model

    if _current_model == model_id:
        log.debug("Model %s already loaded", model_id)
        return False

    if _current_model in EXCLUSIVE_MODELS and model_id in EXCLUSIVE_MODELS:
        old_port = _current_port
        log.info("Flash-Swapping model: %s -> %s", _current_model, model_id)
        _stop_current()
        if old_port:
            _wait_for_port_death(old_port)
        # Bug found via a crash: _stop_current()'s own proc.wait(timeout=20)
        # confirms the OLD PROCESS exited, and _wait_for_port_death above
        # confirms its PORT stopped responding — but on WSL2 (and some
        # native Linux setups under memory pressure), the host reclaiming
        # a just-exited process's mlock'd pages is a SEPARATE, sometimes
        # slower event than the process itself exiting. A crash was traced
        # to exactly this gap: 9B (~9GB resident) was stopped and its port
        # confirmed dead, but the very next model's mlock() of a ~20GB
        # buffer failed with "Cannot allocate memory" because the host
        # hadn't finished reclaiming the prior process's memory yet — and
        # a failed mlock into a not-yet-reclaimed region during active
        # memory pressure is exactly the kind of event that can bring down
        # the whole WSL VM, not just this Python process. Actively wait
        # for available memory to recover before attempting the new load,
        # rather than assuming port-death implies memory-reclaimed.
        _wait_for_memory_available(model_id)
    else:
        log.info("Loading model: %s", model_id)

    _load_model(model_id)
    return True


def _wait_for_port_death(port: int, timeout: int = 10) -> None:
    """Poll heavily until the port is entirely free, ensuring VRAM is released."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if not _is_port_alive(port):
            return
        time.sleep(0.1)
    log.warning("Port %d did not die within %ds, continuing anyway...", port, timeout)


# Rough resident-memory footprint per model — used ONLY as a fallback for
# a model's very FIRST load, before any measured data exists (see
# _measured_model_ram_gb below, which is what's actually used once a
# model has loaded at least once). Deliberately generous (rounds up)
# since under-waiting is what caused the crash this guards against.
# These are ballpark figures from the quant sizes in config/models.yaml,
# NOT measured — expect them to be somewhat off; the measured path is
# the accurate one; this table only covers the cold-start gap before any
# measurement exists.
_APPROX_MODEL_RAM_GB = {
    "9b":         8,
    "27b":        11,
    "27b_ultra":  18,
    "35b":        22,
    "deepcoder":  10,
}

# Where measured footprints are cached across process restarts — a
# fresh Python process has no memory of a prior run's measurements
# otherwise, and re-measuring requires the model to already be loaded
# once, which is exactly the chicken-and-egg this cache avoids repeating
# every restart.
_MEMORY_CACHE_PATH = Path(__file__).parent.parent / "logs" / "_model_ram_measured.json"


def _measured_model_ram_gb(model_id: str) -> Optional[float]:
    """Return this model's actual measured RAM footprint in GB, or None if
    it's never been successfully measured (falls back to the static
    _APPROX_MODEL_RAM_GB table in that case — see _wait_for_memory_available)."""
    try:
        cache = json.loads(_MEMORY_CACHE_PATH.read_text())
        return cache.get(model_id)
    except Exception:
        return None


def _record_measured_model_ram(model_id: str, log_file: Path) -> None:
    """
    Parse model_id's own just-written server log for its ACTUAL resident
    footprint and cache it, so future loads use a real measurement
    instead of the _APPROX_MODEL_RAM_GB guess.

    llama.cpp logs one "... model buffer size = <N> MiB" line per memory
    region it allocated — how many lines and which labels (CUDA0,
    CPU_Mapped, "CPU model buffer size", etc.) appear depends on the
    model's own offload split (see config/models.yaml's partial_offload
    for 27b_ultra, for instance, which deliberately splits across GPU and
    pinned system RAM). Sum ALL such lines from the model's startup block
    — that total is the actual host-memory commitment this function cares
    about, regardless of the GPU/CPU split within it (a value in "CUDA0
    model buffer size" is still memory the host had to make available,
    whether or not it ends up mapped to the GPU device).

    Only reads the LAST such contiguous block in the file (this model's
    most recent startup), same convention clients/model_memory.py already
    uses elsewhere for this log format, per model_manager.py's own
    docstring reference to it — this function doesn't import that module
    (wasn't available to check its exact interface against), but follows
    the same "read the last load block" rule so the two don't disagree
    about which numbers in a multi-restart log file are current.
    """
    try:
        text = log_file.read_text(errors="replace")
    except Exception as e:
        log.debug("Could not read %s to measure RAM for '%s': %s", log_file, model_id, e)
        return

    import re
    pattern = re.compile(r"model buffer size\s*=\s*([\d.]+)\s*MiB")
    matches = pattern.findall(text)
    if not matches:
        log.debug("No 'model buffer size' lines found in %s for '%s' — can't measure", log_file, model_id)
        return

    # "Last load block": take the tail run of matches. Since a fresh
    # ensure_model_loaded() call always writes to a log file that was
    # opened with "a" (append) in _load_model, an OLDER run's lines can
    # still be present above this run's. Rather than parse timestamps,
    # take the last N lines where N is however many buffer-size lines
    # this SAME load produced consecutively at the end of the file — in
    # practice llama.cpp emits all of one load's buffer lines together
    # with nothing but other load_tensors lines between them, so summing
    # every match in the file would double-count prior loads. Taking only
    # matches from the final contiguous group (no more than a handful of
    # non-matching lines between them) is a reasonable middle ground
    # without needing to also parse timestamps or restart markers.
    lines = text.splitlines()
    total_mib = 0.0
    found_any = False
    miss_streak = 0
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            total_mib += float(m.group(1))
            found_any = True
            miss_streak = 0
        else:
            miss_streak += 1
            if found_any and miss_streak > 5:
                break   # ran past this load's contiguous buffer-size block

    if not found_any or total_mib <= 0:
        return

    total_gb = total_mib / 1024.0
    try:
        cache = {}
        if _MEMORY_CACHE_PATH.exists():
            cache = json.loads(_MEMORY_CACHE_PATH.read_text())
        cache[model_id] = round(total_gb, 2)
        _MEMORY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MEMORY_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        log.info("Measured '%s' resident footprint: %.2f GB (cached for future loads)", model_id, total_gb)
    except Exception as e:
        log.warning("Could not cache measured RAM for '%s': %s", model_id, e)


def _wait_for_memory_available(
    incoming_model_id: str, timeout: int = 30, poll_interval: float = 0.5,
) -> None:
    """
    Poll available system memory until there's enough headroom for
    incoming_model_id's footprint, or timeout elapses.

    Prefers a MEASURED footprint (from this model's own prior load — see
    _measured_model_ram_gb/_record_measured_model_ram) over the static
    _APPROX_MODEL_RAM_GB guess table, which only covers the first-ever
    load of a given model before any measurement exists. The static
    table is a rough estimate and may be meaningfully off (it wasn't
    tuned against real logs) — the measured path is the accurate one and
    is what you should expect to be used for anything beyond a model's
    very first load in this project.

    Adds a 10% safety margin on top of whichever figure is used, since
    both the static guess and a single measurement are point estimates,
    not guarantees — llama.cpp's own reported buffer sizes don't include
    every allocation (context/KV cache scales with n_ctx, for instance),
    so treat this as "wait for at least the model weights' worth of
    headroom", not "wait for the exact total the process will ever use".

    Best-effort: if psutil isn't installed, or no size estimate exists
    for this model at all (neither measured nor in the static table),
    falls back to a fixed sleep. Logs a warning rather than raising
    either way — a timeout here means "proceed anyway and let the load
    itself fail/succeed on its own merits", not "block the pipeline
    indefinitely on a machine that's just genuinely tight on RAM".
    """
    measured_gb = _measured_model_ram_gb(incoming_model_id)
    needed_gb   = measured_gb if measured_gb is not None else _APPROX_MODEL_RAM_GB.get(incoming_model_id)
    source      = "measured" if measured_gb is not None else "estimated"

    try:
        import psutil
    except ImportError:
        log.warning(
            "psutil not available — falling back to a fixed 3s delay before "
            "loading %s. Install psutil for an active memory-availability "
            "check instead (see api/server.py's own psutil dependency).",
            incoming_model_id,
        )
        time.sleep(3.0)
        return

    if needed_gb is None:
        log.debug("No RAM estimate for model '%s' — using a fixed 3s delay", incoming_model_id)
        time.sleep(3.0)
        return

    needed_bytes = needed_gb * 1.10 * (1024 ** 3)   # +10% safety margin — see docstring
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        available = psutil.virtual_memory().available
        if available >= needed_bytes:
            log.debug(
                "Memory check: %.1fGB available >= %.1fGB needed (%s, +10%%) for '%s' — proceeding",
                available / (1024 ** 3), needed_bytes / (1024 ** 3), source, incoming_model_id,
            )
            return
        time.sleep(poll_interval)

    log.warning(
        "Only %.1fGB available after %ds waiting for %.1fGB needed (%s) by '%s' — "
        "proceeding anyway (load may fail or the system may come under "
        "memory pressure; consider closing other applications, raising "
        "your .wslconfig memory= limit, or lowering this model's "
        "partial_offload settings in config/models.yaml).",
        psutil.virtual_memory().available / (1024 ** 3), timeout, needed_gb, source, incoming_model_id,
    )


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

    # Measure this model's actual resident footprint from its own
    # just-written log now that health confirms it's genuinely loaded
    # (not just spawned) — see _record_measured_model_ram's docstring.
    # Best-effort: a failure here shouldn't affect the model actually
    # being ready to use, only whether future loads get the accurate
    # (measured) memory-wait figure or fall back to the rougher static
    # estimate.
    try:
        _record_measured_model_ram(model_id, log_file)
    except Exception as e:
        log.warning("Could not measure RAM footprint for '%s': %s", model_id, e)


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


# In _wait_for_health, after getting a 200 from /health, don't return
# immediately — confirm the server actually accepts a real request.
# /health reporting healthy and the server being ready to serve
# /v1/chat/completions without dropping the connection are not
# guaranteed to be the same moment for every llama.cpp version —
# this closes that gap the same way _wait_for_memory_available closed
# the memory-reclaim gap on the stop side.
def _wait_for_health(port, name, proc, timeout=300):
    url = f"http://localhost:{port}/health"
    elapsed = 0
    interval = 3
    log.info("Waiting for %s to load (timeout: %ds)...", name, timeout)
    while elapsed < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} process exited (code {proc.returncode}) before becoming healthy.")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    # Confirm readiness with a trivial real completion,
                    # not just /health — see comment above.
                    if _confirm_completion_ready(port):
                        log.info("%s loaded in %ds", name, elapsed)
                        return
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"{name} did not become healthy within {timeout}s.")

def _confirm_completion_ready(port: int) -> bool:
    """One tiny real completion request, not just /health, to confirm
    the server will actually accept inference calls before we report
    ready and let a pipeline stage race it."""
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=json.dumps({
                "model": "x", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False