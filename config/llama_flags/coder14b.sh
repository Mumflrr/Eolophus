#!/bin/bash
# Qwen2.5 Coder 14B Instruct Q4_K_M — bug fix pass
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
PORT=8085
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting Coder 14B server on port $PORT..."
llama-server -m "$MODEL_PATH" -c 32768 --port $PORT -fa --no-mmap --jinja \
    --host 127.0.0.1 2>&1 | tee "$HOME/pipeline/logs/coder14b_server.log"
