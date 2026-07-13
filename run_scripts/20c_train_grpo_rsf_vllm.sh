#!/usr/bin/env bash
# GRPO + R_sf trainer, vLLM-accelerated rollout generation + lightweight
# HF+PEFT gradient step (rag/train/grpo_rsf_vllm.py). Single GPU.
#
# Why this exists over 20b_train_grpo_rsf_simple.sh: that script generates
# one rollout at a time via HF transformers.generate() — correct, but
# ~17s/rollout serial (measured: 32 rollouts, ~9 min). At design doc D11-14's
# smoke-run scale (100 steps, 5k prompts, G=8 ≈ 6400 rollouts) that's ~30
# hours on one GPU. This script batches rollout generation through vLLM
# (already confirmed both fast and correct for this model) instead, while
# keeping the exact same gradient-step logic as grpo_rsf_simple.py.
#
# UNTESTED on real hardware as of authoring (see grpo_rsf_vllm.py's
# module docstring for known risk areas to check on first run: vLLM's LoRA
# coverage for target_modules="all-linear", max_lora_rank compatibility,
# GPU memory headroom for holding both the vLLM engine and the separate
# training model at once). Strongly do a tiny run first — see Usage below —
# before trusting the full design-doc-scale defaults.
#
# Prerequisites:
#   1. Retriever server running:  bash run_scripts/02_start_retriever.sh
#   2. Training data downloaded (see 20_train_grpo_rsf.sh's header for the
#      download snippet — same train_sf.jsonl file, shared across all three
#      GRPO-RSF scripts).
#   3. Same rag/.venv as grpo_rsf_simple.py — no new dependencies (vLLM was
#      already installed for the Week-1/2 inference pipeline).
#
# Usage:
#   # tiny correctness-only pass FIRST (do this before the full spec):
#   MAX_SAMPLES=20 NUM_EPISODES=3 N_SAMPLES=4 LORA_RANK=16 \
#     bash run_scripts/20c_train_grpo_rsf_vllm.sh
#   # design-doc smoke-run scale, once the above looks sane:
#   bash run_scripts/20c_train_grpo_rsf_vllm.sh

set -euo pipefail

source "$(dirname "$0")/.env_retriever" 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/rag"
MODEL_PATH="$HOME/models/R3-RAG-Qwen"
TRAIN_DATA="$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl"
OUTPUT_DIR="$HOME/logs/grpo_rsf_vllm_$(date +%Y%m%d_%H%M%S)"

# ── Servers (set by .env_retriever or override here) ──────────────────────────
export RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"

# ── Smoke-run knobs (design doc D11-14 defaults; override for a cheap dry run) ─
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
NUM_EPISODES="${NUM_EPISODES:-100}"
N_SAMPLES="${N_SAMPLES:-8}"        # G, rollouts per prompt
BATCH_SIZE="${BATCH_SIZE:-8}"      # questions per episode
LORA_RANK="${LORA_RANK:-64}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.45}"

mkdir -p "$OUTPUT_DIR"

echo "[GRPO-RSF-vLLM] Output dir:  $OUTPUT_DIR"
echo "[GRPO-RSF-vLLM] Retriever:   $RETRIEVE_URL"
echo "[GRPO-RSF-vLLM] max_samples=$MAX_SAMPLES num_episodes=$NUM_EPISODES n_samples=$N_SAMPLES batch_size=$BATCH_SIZE lora_rank=$LORA_RANK vllm_gpu_mem_frac=$VLLM_GPU_MEM_FRAC"

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/train/grpo_rsf_vllm.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$TRAIN_DATA" \
    --save_path "$OUTPUT_DIR/ckpt" \
    --max_samples "$MAX_SAMPLES" \
    --num_episodes "$NUM_EPISODES" \
    --n_samples "$N_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --lora_rank "$LORA_RANK" \
    --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
