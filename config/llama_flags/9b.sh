#!/bin/bash
# 9b.sh — Qwen3.5-9B Q6_K launch. Hot-loaded throughout the pipeline;
# handles classify, plan, critic_a, validate, vision, chess, describe,
# distill, gatekeeper, and short-mode drafting. Full GPU offload — this
# model is small enough to never need partial offload.

source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-9b-q6_k.gguf"
PORT=8081
CTX_LEN=$(read_ctx_len "9b")

require_model_file "$MODEL_PATH"
announce "Qwen3.5-9B" "$PORT"

run_and_log "9b_server.log" \
    -m "$MODEL_PATH" \
    -c "$CTX_LEN" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ngl 99 \
    -fa "$STD_FLASH_ATTN" \
    --jinja \
    -lv 4
