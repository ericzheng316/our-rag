#!/usr/bin/env bash
# Minimal single-GPU GRPO + R_sf trainer — no OpenRLHF/deepspeed/ray/flash-attn
# (rag/train/grpo_rsf_simple.py, attn_implementation="eager").
#
# Use this for a cheap correctness-only check of the turn-level-advantage
# wiring before spending OpenRLHF's distributed-training setup cost
# (20_train_grpo_rsf.sh) on the real smoke run (design doc D11-14: 100 steps,
# 5k prompts, G=8) — that's what 20_train_grpo_rsf.sh is for once this looks
# sane. Defaults here are deliberately tiny (correctness, not throughput).
#
# Prerequisites:
#   1. Retriever server running:  bash run_scripts/02_start_retriever.sh
#   2. Split server running:      bash run_scripts/03_start_split_server.sh
#   3. peft installed in the existing rag/.venv (the only new dependency vs
#      the Week-1/2 inference env — no separate venv needed for this path):
#        source $HOME/rag/.venv/bin/activate && pip install peft
#   4. Training data downloaded (see 20_train_grpo_rsf.sh's header for the
#      download snippet — same train_sf.jsonl file, both scripts share it).
#
# Usage:
#   bash run_scripts/20b_train_grpo_rsf_simple.sh
#   MAX_SAMPLES=100 NUM_EPISODES=20 bash run_scripts/20b_train_grpo_rsf_simple.sh

set -euo pipefail

source "$(dirname "$0")/.env_retriever" 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/rag"
MODEL_PATH="$HOME/models/R3-RAG-Qwen"
TRAIN_DATA="$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl"
OUTPUT_DIR="$HOME/logs/grpo_rsf_simple_$(date +%Y%m%d_%H%M%S)"

# ── Servers (set by .env_retriever or override here) ──────────────────────────
export RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"

# ── Dry-run knobs (tiny by default; bump these once the wiring looks sane) ────
MAX_SAMPLES="${MAX_SAMPLES:-20}"
NUM_EPISODES="${NUM_EPISODES:-2}"
N_SAMPLES="${N_SAMPLES:-4}"    # G, rollouts per prompt
LORA_RANK="${LORA_RANK:-16}"

mkdir -p "$OUTPUT_DIR"

echo "[GRPO-RSF-simple] Output dir:  $OUTPUT_DIR"
echo "[GRPO-RSF-simple] Retriever:   $RETRIEVE_URL"
echo "[GRPO-RSF-simple] max_samples=$MAX_SAMPLES num_episodes=$NUM_EPISODES n_samples=$N_SAMPLES lora_rank=$LORA_RANK"

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/train/grpo_rsf_simple.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$TRAIN_DATA" \
    --save_path "$OUTPUT_DIR/ckpt" \
    --max_samples "$MAX_SAMPLES" \
    --num_episodes "$NUM_EPISODES" \
    --n_samples "$N_SAMPLES" \
    --lora_rank "$LORA_RANK" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
