#!/usr/bin/env bash
# ACEC v7 G6: 100-episode query GRPO with selector/value/belief frozen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
TRAIN_DATA="${TRAIN_DATA:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/train_sf.jsonl}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
ACEC_ARTIFACT_V5="${ACEC_ARTIFACT_V5:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
SELECTOR_ARTIFACT="${SELECTOR_ARTIFACT:?set SELECTOR_ARTIFACT}"
ANSWER_VALUE_ARTIFACT="${ANSWER_VALUE_ARTIFACT:?set ANSWER_VALUE_ARTIFACT}"
SELECTOR_EVAL="${SELECTOR_EVAL:?set SELECTOR_EVAL to selector_heldout_eval.json}"
SELECTED_CALIBRATION="${SELECTED_CALIBRATION:?set SELECTED_CALIBRATION to selected_calibration.json}"
DELTA_C_ABLATION="${DELTA_C_ABLATION:?set DELTA_C_ABLATION to delta_c_ablation.json}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
L="${L:-20}"
K="${K:-5}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SAVE_STEPS="${SAVE_STEPS:-25}"
LR="${LR:-3e-5}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v7_100epi_16x8_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "[v7] TRAIN_GPU and RETRIEVER_GPU must differ" >&2
    exit 2
fi
for path in "$PYTHON" "$MODEL_PATH" "$TRAIN_DATA" "$ACEC_ARTIFACT_V5" "$E5_MODEL_PATH" "$SELECTOR_ARTIFACT" "$ANSWER_VALUE_ARTIFACT" "$SELECTOR_EVAL" "$SELECTED_CALIBRATION" "$DELTA_C_ABLATION"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "[v7] refusing to overwrite: $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" \
    "$REPO_ROOT/rag/train/validate_artifacts_v7.py" \
        --answer_value_artifact "$ANSWER_VALUE_ARTIFACT" \
        --selector_artifact "$SELECTOR_ARTIFACT" \
        --selector_eval "$SELECTOR_EVAL" \
        --selected_calibration "$SELECTED_CALIBRATION" \
        --delta_c_ablation "$DELTA_C_ABLATION" \
        --output "$OUTPUT_ROOT/v7_preflight_gates.json"

MAX_SAMPLES=$((NUM_EPISODES * 16))
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/grpo_rsf_vllm_v7.py" \
        --model_path "$MODEL_PATH" \
        --data_path "$TRAIN_DATA" \
        --save_path "$OUTPUT_ROOT/ckpt" \
        --max_samples "$MAX_SAMPLES" \
        --num_episodes "$NUM_EPISODES" \
        --batch_size 16 \
        --n_samples 8 \
        --n_turns 5 \
        --lr "$LR" \
        --rollout_temperature 0.7 \
        --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC" \
        --max_engine_logprob_mae 0.05 \
        --save_steps "$SAVE_STEPS" \
        --lambda_ans 1.0 \
        --lambda_cov 0.3 \
        --lambda_fmt 0.1 \
        --retrieval_cost 0.05 \
        --kl_coef 0.01 \
        --selector_strategy_v7 learned \
        --selector_artifact_v7 "$SELECTOR_ARTIFACT" \
        --answer_value_artifact_v7 "$ANSWER_VALUE_ARTIFACT" \
        --candidate_pool_size "$L" \
        --selected_k "$K" \
        --coverage_reward_mode legacy_coverage_aux \
        --trace_path "$OUTPUT_ROOT/trajectory_v7.jsonl" \
        --acec_artifact_v5 "$ACEC_ARTIFACT_V5" \
        --e5_model_path "$E5_MODEL_PATH" \
        --acec_nli_model "$NLI_MODEL" \
        2>&1 | tee "$OUTPUT_ROOT/train.log"

echo "[v7] monitor: tail -f '$OUTPUT_ROOT/train.log'"
