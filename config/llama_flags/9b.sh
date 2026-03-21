#!/bin/bash
# 9B Q6_K launch — hot-loaded throughout pipeline
# Small footprint (~6-7 GB); stays in VRAM entire run
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-9b-q6_k.gguf"
PORT=8081
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting 9B server on port $PORT..."
llama-server -m "$MODEL_PATH" -c 32768 --port $PORT -fa --no-mmap --jinja \
    --host 127.0.0.1 2>&1 | tee "$HOME/pipeline/logs/9b_server.log"
