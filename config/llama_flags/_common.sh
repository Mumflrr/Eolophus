#!/bin/bash
# config/llama_flags/_common.sh — shared boilerplate for every model launch script.
#
# Sourced by every {model}.sh script, never run directly.
# Centralizes what should NOT vary between models: conda activation,
# the pattern for reading context_len out of models.yaml, standard flag
# values, and the log-and-run invocation. What genuinely differs per
# model (path, port, offload strategy, thread count) stays in each
# individual script.
#
# Updating llama.cpp itself: this file does NOT hardcode the llama-server
# binary location. It resolves to, in order:
#   1. $LLAMA_SERVER_BIN if set (explicit override)
#   2. `llama-server` on $PATH (default — this is what you have today)
# To point at a different build (e.g. a second checkout for testing a new
# llama.cpp version before trusting it), export LLAMA_SERVER_BIN for that
# shell session rather than editing any script.

set -e

source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

EOLOPHUS_ROOT="$HOME/local-llama/Eolophus"
CONFIG_PATH="$EOLOPHUS_ROOT/config/models.yaml"
LOG_DIR="$EOLOPHUS_ROOT/logs"
mkdir -p "$LOG_DIR"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"

# Standard flag values shared by every model unless a script overrides them.
# -fa 1        Flash attention (all four numeric-boolean flags in llama.cpp
#              accept 1/0, not on/off — standardizing on 1/0 here since
#              that's also what the ultra script already used)
STD_FLASH_ATTN=1

# ── Helpers ──────────────────────────────────────────────────────────────────

# read_ctx_len <model_key>
# Reads context_len for the given models.yaml key. Fails loudly if the
# key doesn't exist rather than silently producing an empty CTX_LEN that
# would make llama-server fall back to a wrong default.
read_ctx_len() {
    local model_key="$1"
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_PATH'))
try:
    print(cfg['models']['$model_key']['context_len'])
except KeyError:
    sys.exit(f\"ERROR: models.yaml has no context_len for '$model_key'\")
"
}

# require_model_file <path>
# Standard existence check with a consistent error message shape across
# every script, instead of four slightly different ad-hoc versions.
require_model_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "ERROR: model file not found at $path"
        echo "Set MODEL_DIR to your models directory, or set MODEL_PATH directly."
        exit 1
    fi
}

# announce <name> <port>
# One consistent startup banner instead of a bare echo per script.
announce() {
    local name="$1"
    local port="$2"
    echo "Starting $name on port $port..."
    echo "  binary: $(command -v "$LLAMA_SERVER_BIN" 2>/dev/null || echo "$LLAMA_SERVER_BIN (not found on PATH — will fail)")"
}

# run_and_log <log_filename> <llama-server args...>
# The tee pattern every script ends with, deduplicated. Takes the log
# filename (not full path — LOG_DIR is prepended) as $1, everything after
# that is passed straight through to llama-server.
run_and_log() {
    local log_file="$1"
    shift
    "$LLAMA_SERVER_BIN" "$@" 2>&1 | tee "$LOG_DIR/$log_file"
}
