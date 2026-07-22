#!/usr/bin/env bash
# Matched 16q x 8-sample outcome-only GRPO ablation on two H100s.
# GPU 0 trains/serves rollouts; the persistent retriever runs on GPU 1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
TRAIN_DATA="${TRAIN_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/train_sf.jsonl}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/outcome_only_100epi_16x8_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

NUM_EPISODES="${NUM_EPISODES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
N_TURNS="${N_TURNS:-5}"
LORA_RANK="${LORA_RANK:-64}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
KL_COEF="${KL_COEF:-0.01}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.60}"
SAVE_STEPS="${SAVE_STEPS:-25}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$TRAIN_DATA" ]]; then
    echo "Missing Python runtime or training data" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify different cards" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT/train.log" || -e "$OUTPUT_ROOT/ckpt" || -e "$OUTPUT_ROOT/run_contract.json" ]]; then
    echo "Refusing to mix runs in non-empty output: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] arm=outcome-only reward=terminal-EM"
echo "[preflight] episodes=$NUM_EPISODES questions=$BATCH_SIZE samples=$N_SAMPLES turns=$N_TURNS"
echo "[preflight] lr=$LEARNING_RATE kl=$KL_COEF temp=$ROLLOUT_TEMPERATURE"
echo "[preflight] train_gpu=$TRAIN_GPU retriever_gpu=$RETRIEVER_GPU vllm=$VLLM_GPU_MEM_FRAC"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_grpo_estimator_v2.py"
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_outcome_only_contract.py"
fi

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["outcome-only retriever health check"]}, timeout=120); r.raise_for_status(); data=r.json(); assert isinstance(data, list) and len(data)==1; print("[preflight] retriever=PASS docs="+str(len(data[0])))'

MAX_SAMPLES=$((NUM_EPISODES * BATCH_SIZE))
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/grpo_rsf_vllm_outcome_only.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$TRAIN_DATA" \
        --save_path "$OUTPUT_ROOT/ckpt" \
        --max_samples "$MAX_SAMPLES" \
        --num_episodes "$NUM_EPISODES" \
        --batch_size "$BATCH_SIZE" \
        --n_samples "$N_SAMPLES" \
        --n_turns "$N_TURNS" \
        --lora_rank "$LORA_RANK" \
        --lr "$LEARNING_RATE" \
        --kl_coef "$KL_COEF" \
        --rollout_temperature "$ROLLOUT_TEMPERATURE" \
        --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
        --max_engine_logprob_mae 0.05 \
        --save_steps "$SAVE_STEPS" \
        2>&1 | tee "$OUTPUT_ROOT/train.log"

echo "Outcome-only log: $OUTPUT_ROOT/train.log"
echo "Outcome-only contract: $OUTPUT_ROOT/run_contract.json"
