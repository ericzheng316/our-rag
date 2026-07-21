#!/usr/bin/env bash
# Matched quick held-out evaluation: frozen R3 vs ACEC-v5 checkpoints.
# The persistent FP32 retriever must already be running on RETRIEVER_GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
DEV_DATA="${DEV_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/dev.jsonl}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$PERSIST_ROOT/logs/acec_v5_100epi_16x8_9d31b7c}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v5_heldout_quick_$(date +%Y%m%d_%H%M%S)}"

MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_TURNS="${N_TURNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-20260721}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.60}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify different cards" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$DEV_DATA" ]]; then
    echo "Missing Python runtime or held-out dataset" >&2
    exit 2
fi
for step in 25 50 75 100; do
    test -f "$TRAIN_RUN_ROOT/ckpt/step_$step/adapter_config.json"
    test -f "$TRAIN_RUN_ROOT/ckpt/step_$step/adapter_model.safetensors"
done
if [[ -e "$OUTPUT_ROOT/results.json" || -e "$OUTPUT_ROOT/eval.log" ]]; then
    echo "Refusing to overwrite prior evaluation: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] heldout=$DEV_DATA questions=$MAX_QUESTIONS seed=$SEED"
echo "[preflight] temperature=$TEMPERATURE batch=$BATCH_SIZE n_turns=$N_TURNS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_eval_r3_lora_checkpoints.py"
fi

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["held-out evaluator health check"]}, timeout=120); r.raise_for_status(); data=r.json(); assert isinstance(data, list) and len(data)==1; print("[preflight] retriever=PASS docs="+str(len(data[0])))'

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/eval_r3_lora_checkpoints.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$DEV_DATA" \
        --output_dir "$OUTPUT_ROOT" \
        --max_questions "$MAX_QUESTIONS" \
        --batch_size "$BATCH_SIZE" \
        --n_turns "$N_TURNS" \
        --temperature "$TEMPERATURE" \
        --seed "$SEED" \
        --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
        --retrieve_url "$RETRIEVE_URL" \
        --adapter "step25=$TRAIN_RUN_ROOT/ckpt/step_25" \
        --adapter "step50=$TRAIN_RUN_ROOT/ckpt/step_50" \
        --adapter "step75=$TRAIN_RUN_ROOT/ckpt/step_75" \
        --adapter "step100=$TRAIN_RUN_ROOT/ckpt/step_100" \
        2>&1 | tee "$OUTPUT_ROOT/eval.log"

echo "Held-out results: $OUTPUT_ROOT/results.json"
