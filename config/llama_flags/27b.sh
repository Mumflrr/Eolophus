#!/bin/bash
source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

# 27B IQ2_S launch — ideation only (long mode)
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-27b-iq2_s.gguf"
CONFIG_PATH="$HOME/local-llama/Eolophus/config/models.yaml"
CTX_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['models']['27b']['context_len'])")
PORT=8082
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting 27B server on port $PORT..."
llama-server -m "$MODEL_PATH" -c $CTX_LEN --port $PORT -fa on --jinja -ngl 99\
    --host 127.0.0.1 2>&1 | tee "$HOME/local-llama/Eolophus/logs/27b_server.log"
