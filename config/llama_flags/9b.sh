#!/bin/bash
source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

# 9B Q6_K launch — hot-loaded throughout pipeline
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-9b-q6_k.gguf"
CONFIG_PATH="$HOME/local-llama/Eolophus/config/models.yaml"
CTX_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['models']['9b']['context_len'])")
PORT=8081
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting 9B server on port $PORT..."

llama-server -m "$MODEL_PATH" -c $CTX_LEN --port $PORT -fa on --jinja -ngl 99 \
    --host 127.0.0.1 2>&1 | tee "$HOME/local-llama/Eolophus/logs/9b_server.log"