#!/bin/bash
# DeepCoder 14B Q4_K_M — appraisal and critic_b
MODEL_PATH="${MODEL_DIR:-$HOME/models}/deepcoder-14b-q4_k_m.gguf"
PORT=8084
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting DeepCoder server on port $PORT..."
llama-server -m "$MODEL_PATH" -c 32768 --port $PORT -fa --no-mmap --jinja \
    --host 127.0.0.1 2>&1 | tee "$HOME/pipeline/logs/deepcoder_server.log"
