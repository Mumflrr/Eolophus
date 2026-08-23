#!/bin/bash
# deepcoder.sh — DeepCoder-14B Q4_K_M launch. Appraisal and critic_b
# roles — RL-trained correctness reasoning. Loaded on demand. Full GPU
# offload.

source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_DIR:-$HOME/models}/deepcoder-14b-q4_k_m.gguf"
PORT=8084
CTX_LEN=$(read_ctx_len "deepcoder")

require_model_file "$MODEL_PATH"
announce "DeepCoder-14B" "$PORT"

run_and_log "deepcoder_server.log" \
    -m "$MODEL_PATH" \
    -c "$CTX_LEN" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ngl 99 \
    -fa "$STD_FLASH_ATTN" \
    --jinja
