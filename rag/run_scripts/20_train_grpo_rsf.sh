#!/usr/bin/env bash
# GRPO + R_sf training script
#
# Prerequisites:
#   1. Retriever server running:  bash run_scripts/02_start_retriever.sh
#   2. Split server running:      bash run_scripts/03_start_split_server.sh
#   3. Training data downloaded:
#        python3 -c "
#        from datasets import load_dataset; import json
#        ds = load_dataset('hotpot_qa', 'fullwiki', split='train')
#        with open('/root/data/flashrag_datasets/hotpotqa/train_sf.jsonl','w') as f:
#            for r in ds:
#                f.write(json.dumps({'id':r['id'],'question':r['question'],
#                  'golden_answers':[r['answer']],'supporting_facts':r['supporting_facts'],
#                  'type':r['type']}) + '\n')
#        "
#
# Usage:
#   bash run_scripts/20_train_grpo_rsf.sh

set -euo pipefail

source "$(dirname "$0")/.env_retriever" 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/rag"
OPENRLHF_DIR="$REPO_ROOT/train/R3RAG_OpenRLHF"
MODEL_PATH="$HOME/models/R3-RAG-Qwen"
TRAIN_DATA="$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl"
OUTPUT_DIR="$HOME/logs/grpo_rsf_$(date +%Y%m%d_%H%M%S)"

# ── Servers (set by .env_retriever or override here) ──────────────────────────
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"
SPLIT_URL="${SPLIT_URL:-http://127.0.0.1:8082/split_query}"

mkdir -p "$OUTPUT_DIR"

echo "[GRPO-RSF] Output dir: $OUTPUT_DIR"
echo "[GRPO-RSF] Retriever:  $RETRIEVE_URL"
echo "[GRPO-RSF] Splitter:   $SPLIT_URL"

# ── Training ───────────────────────────────────────────────────────────────────
cd "$OPENRLHF_DIR"

PYTHONPATH="$OPENRLHF_DIR:$PYTHONPATH" \
"$REPO_ROOT/.venv/bin/python" "$OPENRLHF_DIR/openrlhf/cli/train_grpo_rsf.py" \
    --pretrain "$MODEL_PATH" \
    --save_path "$OUTPUT_DIR/ckpt" \
    --save_steps 50 \
    --logging_steps 1 \
    --eval_steps 50 \
    --micro_train_batch_size 1 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 8 \
    --n_samples_per_prompt 4 \
    --max_epochs 1 \
    --num_episodes 200 \
    --prompt_max_len 2048 \
    --generate_max_len 512 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --advantage_estimator reinforce \
    --prompt_data "$TRAIN_DATA" \
    --prompt_data_probs 1.0 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
