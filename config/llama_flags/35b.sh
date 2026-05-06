#!/bin/bash
# 35B MoE launch config for RTX 3080/4060 (8-10 GB VRAM)
# Target: ~8 GB VRAM via partial expert offload
# Expected throughput: ~30-36 tok/s (The "VRAM Miracle" config)
#
# Flags:
#   --n-cpu-moe 38     The magic number for 8GB cards: offloads just enough experts to CPU
#   -ngl 99            Offload all standard non-expert layers to GPU
#   -ctk q8_0          KV cache keys quantised to 8-bit (VRAM saving)
#   -ctv q8_0          KV cache values quantised to 8-bit
#   -fa 1              Flash Attention (reduces attention VRAM pressure)
#   -b 2048 -ub 2048   Higher batch sizes for much faster prompt processing
#   -c 65536           Context window (tune down to 32768 if VRAM pressure)
#   -t                 Thread count for CPU expert computation

source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama

MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.6-35b-moe-ud-q4_k_m.gguf"
CONFIG_PATH="$HOME/local-llama/Eolophus/config/models.yaml"
CTX_LEN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['models']['35b']['context_len'])")
PORT=8083
THREADS=10  # Adjust: physical_cores / 1.5 rounded down

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Set MODEL_DIR or place model at $HOME/models/"
    exit 1
fi

echo "Starting 35B MoE server on port $PORT..."
echo "Model: $MODEL_PATH"
echo "Threads: $THREADS"

llama-server \
    -m "$MODEL_PATH" \
    -c $CTX_LEN \
    --port $PORT \
    -ot ".ffn_.*_exps.=CPU" \
    -ngl 99 \
    -ctk turbo3 \
    -ctv turbo3 \
    -fa 1 \
    -b 2048 \
    -ub 2048 \
    -t $THREADS \
    --mlock \
    --jinja \
    --host 127.0.0.1 \
    2>&1 | tee "$HOME/local-llama/Eolophus/logs/35b_server.log"