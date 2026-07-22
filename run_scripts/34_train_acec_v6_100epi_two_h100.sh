#!/usr/bin/env bash
# ACEC-v6 100-episode run. Run script 33 first on the same H100 environment.

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
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
ACEC_ARTIFACT_V5="${ACEC_ARTIFACT_V5:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v6_100epi_16x8_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

NUM_EPISODES="${NUM_EPISODES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-8}"
LORA_RANK="${LORA_RANK:-64}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.60}"
SAVE_STEPS="${SAVE_STEPS:-25}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-0}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$TRAIN_DATA" || ! -f "$ACEC_ARTIFACT_V5" ]]; then
    echo "Missing Python, training data, or ACEC-v5 artifact" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify different cards" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT/train.log" || -e "$OUTPUT_ROOT/ckpt" || -e "$OUTPUT_ROOT/trajectory_v6.jsonl" || -e "$OUTPUT_ROOT/v6_contract.json" ]]; then
    echo "Refusing to mix runs in non-empty output: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] arm=acec-v6 episodes=$NUM_EPISODES questions=$BATCH_SIZE samples=$N_SAMPLES"
echo "[preflight] save_steps=$SAVE_STEPS guard=0.50/-0.10 answer_kl=2.0 adaptive_kl=0.08/0.05"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" -c \
    'import sys; from belief.acec.calibration_v5 import load_calibration_artifact_v5; a=load_calibration_artifact_v5(sys.argv[1]); assert a.metrics["gate"]["pass"] is True; print("[preflight] v5 artifact gate=PASS")' \
    "$ACEC_ARTIFACT_V5"

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_acec_v6_contract.py"
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_grpo_estimator_v2.py"
fi

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["ACEC-v6 retriever health check"]}, timeout=120); r.raise_for_status(); data=r.json(); assert isinstance(data, list) and len(data)==1; print("[preflight] retriever=PASS docs="+str(len(data[0])))'

MAX_SAMPLES=$((NUM_EPISODES * BATCH_SIZE))
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/grpo_rsf_vllm_v6.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$TRAIN_DATA" \
        --save_path "$OUTPUT_ROOT/ckpt" \
        --max_samples "$MAX_SAMPLES" \
        --num_episodes "$NUM_EPISODES" \
        --batch_size "$BATCH_SIZE" \
        --n_samples "$N_SAMPLES" \
        --lora_rank "$LORA_RANK" \
        --rollout_temperature 0.7 \
        --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
        --max_engine_logprob_mae 0.05 \
        --save_steps "$SAVE_STEPS" \
        --lambda_ans 1.0 \
        --lambda_cov 0.3 \
        --lambda_fmt 0.1 \
        --retrieval_cost 0.05 \
        --kl_coef 0.01 \
        --answer_guard_coverage 0.50 \
        --premature_answer_penalty 0.10 \
        --answer_kl_multiplier 2.0 \
        --kl_high_watermark 0.08 \
        --kl_recovery_watermark 0.05 \
        --kl_max_coef 0.05 \
        --trace_path "$OUTPUT_ROOT/trajectory_v6.jsonl" \
        --acec_artifact_v5 "$ACEC_ARTIFACT_V5" \
        --e5_model_path "$E5_MODEL_PATH" \
        --acec_nli_model "$NLI_MODEL" \
        2>&1 | tee "$OUTPUT_ROOT/train.log"

echo "ACEC-v6 100-episode log: $OUTPUT_ROOT/train.log"
echo "ACEC-v6 trajectory trace: $OUTPUT_ROOT/trajectory_v6.jsonl"
