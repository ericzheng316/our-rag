#!/usr/bin/env bash
# Fixed held-out comparison across frozen R3, outcome-only, ACEC-v5, ACEC-v6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
DEV_DATA="${DEV_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/dev.jsonl}"
OUTCOME_RUN_ROOT="${OUTCOME_RUN_ROOT:-$PERSIST_ROOT/logs/outcome_only_100epi_16x8_9cfc9ce}"
V5_RUN_ROOT="${V5_RUN_ROOT:-$PERSIST_ROOT/logs/acec_v5_100epi_16x8_9d31b7c}"
V6_RUN_ROOT="${V6_RUN_ROOT:-$PERSIST_ROOT/logs/acec_v6_100epi_16x8_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/v6_v5_outcome_heldout_$(date +%Y%m%d_%H%M%S)}"

MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-20260721}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.60}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify different cards" >&2
    exit 2
fi
for path in \
    "$OUTCOME_RUN_ROOT/ckpt/step_50" "$OUTCOME_RUN_ROOT/ckpt/step_100" \
    "$V5_RUN_ROOT/ckpt/step_50" "$V5_RUN_ROOT/ckpt/step_100" \
    "$V6_RUN_ROOT/ckpt/step_50" "$V6_RUN_ROOT/ckpt/step_100"; do
    test -f "$path/adapter_config.json"
    test -f "$path/adapter_model.safetensors"
done
if [[ -e "$OUTPUT_ROOT/results.json" || -e "$OUTPUT_ROOT/eval.log" ]]; then
    echo "Refusing to overwrite prior evaluation: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] questions=$MAX_QUESTIONS seed=$SEED temperature=0"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["v6 held-out health check"]}, timeout=120); r.raise_for_status(); print("[preflight] retriever=PASS")'

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/eval_r3_lora_checkpoints.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$DEV_DATA" \
        --output_dir "$OUTPUT_ROOT" \
        --max_questions "$MAX_QUESTIONS" \
        --batch_size "$BATCH_SIZE" \
        --n_turns 5 \
        --temperature 0 \
        --seed "$SEED" \
        --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
        --retrieve_url "$RETRIEVE_URL" \
        --adapter "outcome_step50=$OUTCOME_RUN_ROOT/ckpt/step_50" \
        --adapter "outcome_step100=$OUTCOME_RUN_ROOT/ckpt/step_100" \
        --adapter "v5_step50=$V5_RUN_ROOT/ckpt/step_50" \
        --adapter "v5_step100=$V5_RUN_ROOT/ckpt/step_100" \
        --adapter "v6_step50=$V6_RUN_ROOT/ckpt/step_50" \
        --adapter "v6_step100=$V6_RUN_ROOT/ckpt/step_100" \
        2>&1 | tee "$OUTPUT_ROOT/eval.log"

echo "V6/V5/outcome held-out results: $OUTPUT_ROOT/results.json"
