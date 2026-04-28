#!/bin/bash
source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

# Qwen2.5 Coder 14B Instruct Q4_K_M — bug fix pass
MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
CONFIG_PATH="$HOME/local-llama/Eolophus/config/models.yaml"
CTX_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['models']['coder14b']['context_len'])")
PORT=8085
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting Coder 14B server on port $PORT..."
llama-server -m "$MODEL_PATH" -c $CTX_LEN --port $PORT -fa on --jinja -ngl 99\
    --host 127.0.0.1 2>&1 | tee "$HOME/local-llama/Eolophus/logs/coder14b_server.log"
