#!/bin/bash
# 35b.sh — Qwen3.6-35B MoE UD-Q4_K_M launch. Long-mode drafting and
# complex synthesis. Partial expert offload — this is the one model in
# the standard fleet that doesn't fit in 8-10GB VRAM whole, so unlike
# the other three scripts, this one genuinely needs different flags:
# tensor-level expert offload to CPU, quantized KV cache, and tuned
# batch/thread sizes. Do not "simplify" this to match 9b/27b/deepcoder —
# the difference here is real, not accidental duplication.

source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.6-35b-moe-ud-q4_k_m.gguf"
PORT=8083
CTX_LEN=$(read_ctx_len "35b")
THREADS="${LLAMA_35B_THREADS:-10}"   # physical_cores / 1.5, rounded down

require_model_file "$MODEL_PATH"
announce "Qwen3.6-35B-MoE" "$PORT"
echo "  threads: $THREADS (override with LLAMA_35B_THREADS)"

run_and_log "35b_server.log" \
    -m "$MODEL_PATH" \
    -c "$CTX_LEN" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ot ".ffn_.*_exps.=CPU" \
    -ngl 99 \
    -fa "$STD_FLASH_ATTN" \
    -b 2048 \
    -ub 2048 \
    -t "$THREADS" \
    --mlock \
    --jinja \
    -lv 4
