#!/usr/bin/env bash
# GRPO + R_sf training script
#
# Prerequisites:
#   1. Retriever server running:  bash run_scripts/02_start_retriever.sh
#      (split server / 03_start_split_server.sh is NOT needed by this script —
#      experience_maker_grpo_rsf.py reads SPLIT_URL but never calls it; that's
#      a leftover from inference_new.py's prerequisites, not this path's)
#   2. Training data downloaded (dataset id needs the hotpotqa/ namespace —
#      bare 'hotpot_qa' 404s on current datasets/huggingface_hub versions):
#        python3 -c "
#        from datasets import load_dataset; import json
#        ds = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', split='train')
#        with open('$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl','w') as f:
#            for r in ds:
#                f.write(json.dumps({'id':r['id'],'question':r['question'],
#                  'golden_answers':[r['answer']],'supporting_facts':r['supporting_facts'],
#                  'type':r['type']}) + '\n')
#        "
#
# Usage:
#   bash run_scripts/20_train_grpo_rsf.sh
#
# Sizing (design doc Section 9, D11-14 smoke run: 100 steps, 5k prompts, G=8):
#   defaults below match that spec. Before spending a full smoke run's compute,
#   do a cheap correctness-only dry run first (does it crash, do avg_R_total /
#   avg_turns in the log look sane) with a tiny override, e.g.:
#     MAX_SAMPLES=20 NUM_EPISODES=2 bash run_scripts/20_train_grpo_rsf.sh
#   This is ablation (b) only (gold-SF reward, no ACEC belief) — ablation (b)
#   is deliberately run before wiring the belief-based R_cov reward (design
#   doc Section 8, risk #1: "run (b) first... most dangerous row in the
#   paper" — if gold-SF alone already captures most of the value, that bounds
#   how much the belief-specific investment is worth before spending on it).

set -euo pipefail

source "$(dirname "$0")/.env_retriever" 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/rag"
OPENRLHF_DIR="$REPO_ROOT/train/R3RAG_OpenRLHF"
MODEL_PATH="$HOME/models/R3-RAG-Qwen"
TRAIN_DATA="$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl"
OUTPUT_DIR="$HOME/logs/grpo_rsf_$(date +%Y%m%d_%H%M%S)"
# Separate venv from rag/.venv on purpose: OpenRLHF's requirements.txt pins
# deepspeed/transformers/ray/flash-attn versions that would otherwise fight
# the already-calibrated Week-1/2 inference venv (vllm + sentence-transformers).
# Setup — flash-attn's setup.py imports torch at build time, so it must be
# installed *before* flash-attn, and built with --no-build-isolation so the
# build sees it (plain `pip install -r requirements.txt` fails on this: pip's
# isolated build env for flash-attn doesn't have torch yet even though torch
# is listed later in the same file):
#   python3 -m venv "$REPO_ROOT/.venv_train" && source "$REPO_ROOT/.venv_train/bin/activate"
#   pip install torch
#   pip install flash-attn==2.7.0.post2 --no-build-isolation   # add MAX_JOBS=4 if it OOMs compiling
#   pip install -r "$OPENRLHF_DIR/requirements.txt"             # rest of the deps; torch/flash-attn already satisfied
#   pip install -e "$OPENRLHF_DIR"
# If flash-attn's from-source build is a blocker for a quick correctness-only
# dry run, set USE_FLASH_ATTN=0 (see below) and skip installing it for now —
# it's only needed for the real smoke run's speed/memory, not correctness.
TRAIN_VENV="${TRAIN_VENV:-$REPO_ROOT/.venv_train}"

# ── Servers (set by .env_retriever or override here) ──────────────────────────
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"
SPLIT_URL="${SPLIT_URL:-http://127.0.0.1:8082/split_query}"

# ── Smoke-run knobs (design doc D11-14 defaults; override for a cheap dry run) ─
LORA_RANK="${LORA_RANK:-64}"
N_SAMPLES="${N_SAMPLES:-8}"        # G, rollouts per prompt
MAX_SAMPLES="${MAX_SAMPLES:-5000}" # prompt pool size
NUM_EPISODES="${NUM_EPISODES:-100}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}" # set 0 to skip (not installed yet / correctness-only dry run)

mkdir -p "$OUTPUT_DIR"

echo "[GRPO-RSF] Output dir:   $OUTPUT_DIR"
echo "[GRPO-RSF] Retriever:    $RETRIEVE_URL"
echo "[GRPO-RSF] Splitter:     $SPLIT_URL"
echo "[GRPO-RSF] lora_rank=$LORA_RANK n_samples_per_prompt=$N_SAMPLES max_samples=$MAX_SAMPLES num_episodes=$NUM_EPISODES use_flash_attn=$USE_FLASH_ATTN"

# ── Training ───────────────────────────────────────────────────────────────────
cd "$OPENRLHF_DIR"

args=(
    --pretrain "$MODEL_PATH"
    --save_path "$OUTPUT_DIR/ckpt"
    --save_steps 50
    --logging_steps 1
    --eval_steps 50
    --micro_train_batch_size 1
    --train_batch_size 8
    --micro_rollout_batch_size 1
    --rollout_batch_size 8
    --n_samples_per_prompt "$N_SAMPLES"
    --max_samples "$MAX_SAMPLES"
    --max_epochs 1
    --num_episodes "$NUM_EPISODES"
    --prompt_max_len 2048
    --generate_max_len 512
    --zero_stage 2
    --bf16
    --gradient_checkpointing
    --lora_rank "$LORA_RANK"
    --lora_alpha $((LORA_RANK * 2))
    --actor_learning_rate 5e-7
    --init_kl_coef 0.01
    --advantage_estimator reinforce
    --prompt_data "$TRAIN_DATA"
    --prompt_data_probs 1.0
)
if [[ "$USE_FLASH_ATTN" == "1" ]]; then
    args+=(--flash_attn)
fi

PYTHONPATH="$OPENRLHF_DIR:$PYTHONPATH" \
"$TRAIN_VENV/bin/python" "$OPENRLHF_DIR/openrlhf/cli/train_grpo_rsf.py" \
    "${args[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
