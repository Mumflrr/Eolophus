#!/bin/bash
# 35B MoE launch config for RTX 3080 (10 GB VRAM)
# Target: ~8 GB VRAM via full expert offload to CPU RAM
# Expected throughput: 8-15 tok/s depending on RAM bandwidth
#
# Flags:
#   -ot "exps=CPU"     MoE expert tensors streamed from RAM (not VRAM)
#   -ctk q8_0          KV cache keys quantised to 8-bit (VRAM saving)
#   -ctv q8_0          KV cache values quantised to 8-bit
#   -fa on             Flash Attention (reduces attention VRAM pressure)
#   --no-mmap          Load full model into RAM (consistent inference speed)
#   --fit              Auto-fit as many layers as possible into VRAM
#   -c 65536           Context window (tune down to 32768 if VRAM pressure)
#   --ctx-shift        Shift context window when full (prevents hard failure)
#   -t                 Thread count for CPU expert computation
#                      Start at physical_cores / 1.5, sweep from there
#
# To find physical core count:
#   nproc --all   (logical)
#   lscpu | grep "Core(s) per socket"

source /home/dgart/miniconda3/etc/profile.d/conda.sh
conda activate llama


MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.5-35b-moe-ud-q4_k_xl.gguf"
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
    -ot "exps=CPU" \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa on \
    --no-mmap \
    --fit on \
    -t $THREADS \
    --jinja \
    --host 127.0.0.1 \
    2>&1 | tee "$HOME/local-llama/Eolophus/logs/35b_server.log"
