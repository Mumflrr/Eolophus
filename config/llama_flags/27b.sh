#!/bin/bash
# 27B IQ2_S launch — ideation only (long mode)
# Load on demand; ~7-8 GB
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-27b-iq2_s.gguf"
PORT=8082
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting 27B server on port $PORT..."
llama-server -m "$MODEL_PATH" -c 16384 --port $PORT -fa --no-mmap --jinja \
    --host 127.0.0.1 2>&1 | tee "$HOME/pipeline/logs/27b_server.log"
