#!/bin/bash
source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

# DeepCoder 14B Q4_K_M — appraisal and critic_b
MODEL_PATH="${MODEL_DIR:-$HOME/models}/deepcoder-14b-q4_k_m.gguf"
CONFIG_PATH="$HOME/local-llama/Eolophus/config/models.yaml"
CTX_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['models']['deepcoder']['context_len'])")
PORT=8084
if [ ! -f "$MODEL_PATH" ]; then echo "ERROR: $MODEL_PATH not found"; exit 1; fi
echo "Starting DeepCoder server on port $PORT..."
llama-server -m "$MODEL_PATH" -c $CTX_LEN --port $PORT -fa on --jinja -ngl 99\
    --host 127.0.0.1 2>&1 | tee "$HOME/local-llama/Eolophus/logs/deepcoder_server.log"
