"""
clients/model_memory.py — parse llama.cpp's -lv 4 load-time memory logging.

Requires the model's launch script to pass -lv 4 (verbosity 4) — without
it, llama.cpp's server prints only "model loaded" / "listening on ..." with
no buffer-size breakdown at all (confirmed empirically: default verbosity
produces nothing to parse here, see the -lv 4 addition to 9b.sh). Add -lv 4
to 27b.sh/35b.sh too if you want memory data for those roles as well —
this module works off any model's log file the same way, but only once
that script is emitting the breakdown lines in the first place.

Parses the per-buffer "size = ... MiB" lines that appear once, right after
load_tensors starts, rather than the "memory breakdown" summary table
(common_memory_breakdown_print) — the table uses fixed-width column
alignment that's a more fragile regex target than the individual labelled
lines, and the individual lines reflect actual post-allocation buffer
sizes rather than the table's pre-flight projection (they agreed almost
exactly in the one real log sample this was built against, but the
buffer-size lines are the ones that stay correct if projection and
reality ever diverge under memory pressure).

Lines this looks for (CUDA0 device only — matches the single-GPU setup
these scripts assume; a multi-GPU rig would need this extended to sum
across CUDA0/CUDA1/etc, not something this module currently does):

    load_tensors:        CUDA0 model buffer size =  6498.63 MiB
    llama_kv_cache:      CUDA0 KV buffer size =  1024.00 MiB
    llama_memory_recurrent:      CUDA0 RS buffer size =   201.00 MiB
    sched_reserve:      CUDA0 compute buffer size =   120.02 MiB

Also captures the CPU-side lines for completeness:
    load_tensors:   CPU_Mapped model buffer size =   795.70 MiB
    sched_reserve:  CUDA_Host compute buffer size =    48.02 MiB
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


# Each tuple: (regex, field name on ModelMemoryUsage). All match a
# "... size = <float> MiB" line; group(1) is the number.
_CUDA_PATTERNS = [
    (re.compile(r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB"),            "model_mib"),
    (re.compile(r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB"),               "kv_cache_mib"),
    (re.compile(r"CUDA0 RS buffer size\s*=\s*([\d.]+)\s*MiB"),               "recurrent_mib"),
    (re.compile(r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB"),          "compute_mib"),
]
_HOST_PATTERNS = [
    (re.compile(r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB"),       "host_model_mib"),
    (re.compile(r"CUDA_Host compute buffer size\s*=\s*([\d.]+)\s*MiB"),      "host_compute_mib"),
]

# "listening on http://..." marks the end of one load's log output — used
# to find the LAST load block in an appended (not truncated) log file, so
# a server that's been flash-swapped several times doesn't get an earlier
# load's numbers mixed with the current one.
_LISTENING_RE = re.compile(r"listening on http://")


class ModelMemoryUsage(BaseModel):
    model_mib:        Optional[float] = None   # weights, GPU (CUDA0)
    kv_cache_mib:      Optional[float] = None   # KV cache, GPU
    recurrent_mib:     Optional[float] = None   # recurrent/SSM state buffer, GPU (0 for non-hybrid archs)
    compute_mib:       Optional[float] = None   # compute/scratch buffer, GPU
    host_model_mib:    Optional[float] = None   # weights portion mapped to host RAM (CPU_Mapped)
    host_compute_mib:  Optional[float] = None   # compute buffer, host RAM

    @property
    def gpu_total_mib(self) -> Optional[float]:
        """Sum of all CUDA0 buffers actually captured. None if none were found."""
        parts = [self.model_mib, self.kv_cache_mib, self.recurrent_mib, self.compute_mib]
        found = [p for p in parts if p is not None]
        return sum(found) if found else None

    @property
    def host_total_mib(self) -> Optional[float]:
        parts = [self.host_model_mib, self.host_compute_mib]
        found = [p for p in parts if p is not None]
        return sum(found) if found else None


def parse_load_memory(log_text: str) -> Optional[ModelMemoryUsage]:
    """
    Parse the LAST load block in log_text (see _LISTENING_RE) for memory
    buffer lines. Returns None if no buffer-size lines were found at all
    (e.g. -lv 4 wasn't set for this launch, or the block never reached
    the "listening on" line because the load crashed/hasn't finished).

    Returns a ModelMemoryUsage with whichever fields WERE found even if
    others are missing — a recurrent-architecture model with SSM state
    (like Qwen3.5's hybrid attention, see the RS buffer line) will have
    recurrent_mib populated; a plain transformer might not have that line
    at all, and that's fine, it just stays None rather than being forced
    to 0 (0 would incorrectly claim "measured and confirmed zero" for
    something that was never measured).
    """
    # Isolate the last load block: everything after the last "listening on"
    # line is post-load traffic, not useful here; everything between the
    # second-to-last and last "listening on" (or start of file, if only
    # one load happened) is the load block we want.
    listen_matches = list(_LISTENING_RE.finditer(log_text))
    if not listen_matches:
        log.warning("No 'listening on' marker found in log — log may be incomplete or truncated mid-load")
        return None

    last_listen_end = listen_matches[-1].end()
    block_start = listen_matches[-2].end() if len(listen_matches) >= 2 else 0
    block = log_text[block_start:last_listen_end]

    result: dict = {}
    for pattern, field in _CUDA_PATTERNS + _HOST_PATTERNS:
        m = pattern.search(block)
        if m:
            result[field] = float(m.group(1))

    if not result:
        log.info(
            "No memory buffer-size lines found in load block — likely -lv 4 "
            "was not set for this launch (see this module's docstring)."
        )
        return None

    return ModelMemoryUsage(**result)


def get_load_memory(model_id: str, logs_dir: Path) -> Optional[ModelMemoryUsage]:
    """
    Convenience wrapper: read {logs_dir}/{model_id}_server.log (the exact
    path model_manager.py's _load_model() writes to) and parse it.

    Returns None (not an exception) on any failure — missing file, empty
    file, no parseable lines — matching search_web()'s established
    best-effort philosophy elsewhere in this codebase: memory data is a
    nice-to-have for the UI, not something that should be able to break
    model loading if the log format ever shifts under a llama.cpp update.
    """
    log_path = logs_dir / f"{model_id}_server.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log.info("No log file at %s — skipping memory capture", log_path)
        return None
    except Exception as e:
        log.warning("Failed to read %s for memory capture: %s", log_path, e)
        return None

    return parse_load_memory(text)