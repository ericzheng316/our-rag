#!/usr/bin/env bash
# Isolated ACEC-v4 smoke/pilot.  The persistent retriever must already be
# serving on RETRIEVER_GPU; this script uses TRAIN_GPU for vLLM + HF/PEFT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
TRAIN_DATA="${TRAIN_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/train_sf.jsonl}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
ACEC_ARTIFACT_V4="${ACEC_ARTIFACT_V4:-$PERSIST_ROOT/logs/acec_calibration_v4/acec_calibration_v4.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v4_validation_$(date +%Y%m%d_%H%M%S)}"

NUM_EPISODES="${NUM_EPISODES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
LORA_RANK="${LORA_RANK:-64}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
VLLM_FRACTION="${VLLM_FRACTION:-0.60}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python runtime not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Training data not found: $TRAIN_DATA" >&2
    exit 2
fi
if [[ ! -f "$ACEC_ARTIFACT_V4" ]]; then
    echo "ACEC-v4 artifact not found: $ACEC_ARTIFACT_V4" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify different cards" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] episodes=$NUM_EPISODES questions=$BATCH_SIZE samples=$N_SAMPLES"
echo "[preflight] train_gpu=$TRAIN_GPU retriever_gpu=$RETRIEVER_GPU vllm=$VLLM_FRACTION"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    PYTHONPATH="$REPO_ROOT/rag/src" \
        "$PYTHON" "$REPO_ROOT/rag/src/belief/acec/tests/test_calibration_v4.py"
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_grpo_estimator_v2.py"
fi

MAX_SAMPLES=$((NUM_EPISODES * BATCH_SIZE))
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/grpo_rsf_vllm.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$TRAIN_DATA" \
        --save_path "$OUTPUT_ROOT/ckpt" \
        --max_samples "$MAX_SAMPLES" \
        --num_episodes "$NUM_EPISODES" \
        --batch_size "$BATCH_SIZE" \
        --n_samples "$N_SAMPLES" \
        --lora_rank "$LORA_RANK" \
        --rollout_temperature "$ROLLOUT_TEMPERATURE" \
        --vllm_gpu_mem_frac "$VLLM_FRACTION" \
        --max_engine_logprob_mae 0.05 \
        --save_steps 999999 \
        --lambda_ans 1.0 \
        --lambda_cov 0.3 \
        --lambda_fmt 0.1 \
        --retrieval_cost 0.05 \
        --clip_eps 0.2 \
        --use_acec \
        --acec_artifact_v4 "$ACEC_ARTIFACT_V4" \
        --acec_k_mode fixed \
        --acec_fixed_k 2 \
        2>&1 | tee "$OUTPUT_ROOT/train.log"

echo "ACEC-v4 validation log: $OUTPUT_ROOT/train.log"
