#!/usr/bin/env bash
# One-engine fixed held-out comparison: frozen R3, outcome-only 50/100,
# and the already-trained ACEC-v5 50/100 checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
DEV_DATA="${DEV_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/dev.jsonl}"
OUTCOME_RUN_ROOT="${OUTCOME_RUN_ROOT:-$PERSIST_ROOT/logs/outcome_only_100epi_16x8_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
ACEC_RUN_ROOT="${ACEC_RUN_ROOT:-$PERSIST_ROOT/logs/acec_v5_100epi_16x8_9d31b7c}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_vs_outcome_heldout_$(date +%Y%m%d_%H%M%S)}"

MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_TURNS="${N_TURNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.0}"
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
if [[ ! -x "$PYTHON" || ! -f "$DEV_DATA" ]]; then
    echo "Missing Python runtime or held-out dataset" >&2
    exit 2
fi
for path in \
    "$OUTCOME_RUN_ROOT/ckpt/step_50" \
    "$OUTCOME_RUN_ROOT/ckpt/step_100" \
    "$ACEC_RUN_ROOT/ckpt/step_50" \
    "$ACEC_RUN_ROOT/ckpt/step_100"; do
    test -f "$path/adapter_config.json"
    test -f "$path/adapter_model.safetensors"
done
if [[ -e "$OUTPUT_ROOT/results.json" || -e "$OUTPUT_ROOT/eval.log" ]]; then
    echo "Refusing to overwrite prior evaluation: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] outcome=$OUTCOME_RUN_ROOT"
echo "[preflight] acec=$ACEC_RUN_ROOT"
echo "[preflight] heldout=$DEV_DATA questions=$MAX_QUESTIONS seed=$SEED temperature=$TEMPERATURE"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

PYTHONPATH="$REPO_ROOT/rag/train" \
    "$PYTHON" "$REPO_ROOT/rag/train/tests/test_eval_r3_lora_checkpoints.py"

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["matched ablation evaluator health check"]}, timeout=120); r.raise_for_status(); data=r.json(); assert isinstance(data, list) and len(data)==1; print("[preflight] retriever=PASS docs="+str(len(data[0])))'

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
        --adapter "outcome_step50=$OUTCOME_RUN_ROOT/ckpt/step_50" \
        --adapter "outcome_step100=$OUTCOME_RUN_ROOT/ckpt/step_100" \
        --adapter "acec_step50=$ACEC_RUN_ROOT/ckpt/step_50" \
        --adapter "acec_step100=$ACEC_RUN_ROOT/ckpt/step_100" \
        2>&1 | tee "$OUTPUT_ROOT/eval.log"

echo "Matched ACEC/outcome-only results: $OUTPUT_ROOT/results.json"
