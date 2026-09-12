#!/bin/bash
# 27b.sh — Qwen3.5-27B IQ2_XXS launch. Ideation only (long mode). Loaded
# on demand, unloaded after ideation stage. Full GPU offload — the
# aggressive quant keeps this small enough despite the parameter count.
#
# NOT the same model as 27b_ultra.sh — that's a separate MTP-capable
# GGUF for manually-triggered overnight deep-thinking, partial offload,
# unlimited budget. This script stays fast and full-GPU for its role in
# the normal pipeline flow. See 27b_ultra.sh's header for why both exist.

source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-27b-iq2_xxs.gguf"
PORT=8082
CTX_LEN=$(read_ctx_len "27b")

require_model_file "$MODEL_PATH"
announce "Qwen3.5-27B" "$PORT"

run_and_log "27b_server.log" \
    -m "$MODEL_PATH" \
    -c "$CTX_LEN" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ngl 99 \
    -fa "$STD_FLASH_ATTN" \
    --jinja \
    -lv 4
