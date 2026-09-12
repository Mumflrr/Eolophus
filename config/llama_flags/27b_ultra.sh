#!/bin/bash
# 27b_ultra.sh — Qwen3.6-27B MTP overnight / ultra-mode launch.
#
# Designed for:
#   - Manually triggered overnight deep-thinking runs
#   - Partial VRAM offload: early layers hot in GPU, rest in system RAM
#   - MTP speculative decoding for ~1.7x throughput improvement
#   - Near-unlimited thinking budget (controlled by pipeline, not server)
#
# Separate from 27b.sh (ideation model) intentionally — different GGUF,
# different offload strategy, different role. See 27b.sh's header.
#
# REQUIREMENTS:
#   1. MTP-capable GGUF — NOT the standard 27B file.
#      Download from one of:
#        havenoammo/Qwen3.6-27B-MTP-UD-GGUF   (Unsloth Dynamic XL base)
#        froggeric/Qwen3.6-27B-MTP-GGUF        (Q5_K_M MTP)
#      Place at $MODEL_DIR/qwen3.6-27b-mtp-q4_k_m.gguf (or set MODEL_PATH)
#
#   2. llama.cpp built from PR #22673 (MTP support):
#        git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-mtp
#        cd llama.cpp-mtp
#        git fetch origin pull/22673/head:mtp-pr
#        git checkout mtp-pr
#        cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#        cmake --build build --target llama-server -j$(nproc)
#      Standard llama.cpp builds silently ignore --spec-type mtp.
#      Point this script at that build specifically:
#        LLAMA_SERVER_BIN=/path/to/llama.cpp-mtp/build/bin/llama-server ./27b_ultra.sh
#      (the other four scripts continue using whatever's on $PATH — only
#      this one needs the separate build, see _common.sh's header)
#
# VRAM STRATEGY (8-10 GB card):
#   Qwen3.6-27B Q4_K_M GGUF ≈ 15.6 GB total
#   GPU: early attention layers (dense, high reuse) → fastest path
#   RAM: remaining layers served from pinned system RAM via mmap
#
#   Tuning -ngl:
#     -ngl 20  → ~6 GB VRAM   (conservative, ~8-10 tok/s generation)
#     -ngl 28  → ~8 GB VRAM   (recommended for 8GB cards, the default here)
#     -ngl 32  → ~9.5 GB VRAM (push it if nothing else is using VRAM)
#   Reduce -ngl if llama-server OOMs. Increase for more speed.
#   The 9B must be stopped first (model_manager handles this).
#
# MTP FLAGS:
#   --spec-type mtp       enables MTP speculative decoding (PR #22673)
#   --spec-draft-n-max 3  draft 3 tokens per step — best acceptance/speed
#                         tradeoff; draft 5 adds overhead that eats the gain
#   --parallel 1          MTP requires single-slot — architecture constraint
#
# PERFORMANCE EXPECTATIONS (8-10GB VRAM, ~28 layers on GPU):
#   Generation: ~8-15 tok/s without MTP, ~14-25 tok/s with MTP.
#   This is intentionally an overnight mode. Speed is secondary to depth.

source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_DIR:-$HOME/models}/qwen3.6-27b-mtp-q4_k_m.gguf"
PORT=8085
CTX_LEN=$(read_ctx_len "27b_ultra")

NGL="${ULTRA_NGL:-28}"
MTP_DRAFT="${ULTRA_MTP_DRAFT:-3}"
THREADS="${ULTRA_THREADS:-10}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: MTP model not found at $MODEL_PATH"
    echo ""
    echo "This mode requires an MTP-capable GGUF, NOT the standard 27B file."
    echo "Download from HuggingFace:"
    echo "  havenoammo/Qwen3.6-27B-MTP-UD-GGUF"
    echo "  froggeric/Qwen3.6-27B-MTP-GGUF"
    echo ""
    echo "Set MODEL_DIR to your models directory, or set MODEL_PATH directly."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           ULTRA MODE — Qwen3.6-27B MTP                  ║"
echo "║           Overnight deep-thinking configuration          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Model:      $(basename "$MODEL_PATH")"
echo "║  Port:       $PORT"
echo "║  Binary:     $(command -v "$LLAMA_SERVER_BIN" 2>/dev/null || echo "$LLAMA_SERVER_BIN (not found on PATH — will fail)")"
echo "║  GPU layers: $NGL (remaining in system RAM)"
echo "║  MTP draft:  $MTP_DRAFT tokens/step"
echo "║  Context:    $CTX_LEN tokens"
echo "║  Threads:    $THREADS (CPU expert compute)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "NOTE: MTP requires llama.cpp built from PR #22673 — see header comment"
echo "      for how to point LLAMA_SERVER_BIN at that build specifically."
echo ""

run_and_log "27b_ultra_server.log" \
    -m "$MODEL_PATH" \
    -c "$CTX_LEN" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ngl "$NGL" \
    --spec-type mtp \
    --spec-draft-n-max "$MTP_DRAFT" \
    --parallel 1 \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa "$STD_FLASH_ATTN" \
    -b 2048 \
    -ub 2048 \
    -t "$THREADS" \
    --mlock \
    --jinja \
    -lv 4
