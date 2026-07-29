#!/usr/bin/env bash
# ACEC v7 G4: frozen checkpoint + frozen selector end-to-end inference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
ADAPTER="${ADAPTER:-}"
DATA_PATH="${DATA_PATH:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/dev_distractor.jsonl}"
SELECTOR_ARTIFACT="${SELECTOR_ARTIFACT:?set SELECTOR_ARTIFACT}"
ANSWER_VALUE_ARTIFACT="${ANSWER_VALUE_ARTIFACT:?set ANSWER_VALUE_ARTIFACT}"
ACEC_ARTIFACT_V5="${ACEC_ARTIFACT_V5:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
INFERENCE_GPU="${INFERENCE_GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/logs/acec_v7_frozen_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-16}"
L="${L:-20}"
K="${K:-5}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$MODEL_PATH" "$DATA_PATH" "$SELECTOR_ARTIFACT" "$ANSWER_VALUE_ARTIFACT" "$ACEC_ARTIFACT_V5" "$E5_MODEL_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done
if [[ -n "$ADAPTER" && ! -e "$ADAPTER" ]]; then
    echo "[v7] adapter does not exist: $ADAPTER" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[v7] refusing to overwrite: $OUTPUT_DIR" >&2
    exit 2
fi

adapter_args=()
if [[ -n "$ADAPTER" ]]; then
    adapter_args=(--adapter "$ADAPTER")
fi

PYTHONPATH="$REPO_ROOT/rag/src:$REPO_ROOT/rag/train" \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$INFERENCE_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/infer_frozen_v7.py" \
        --model_path "$MODEL_PATH" \
        "${adapter_args[@]}" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --max_questions "$MAX_QUESTIONS" \
        --question_batch_size "$QUESTION_BATCH_SIZE" \
        --n_samples 1 \
        --temperature 0.0 \
        --candidate_pool_size "$L" \
        --selected_k "$K" \
        --selector_strategy learned \
        --selector_artifact "$SELECTOR_ARTIFACT" \
        --answer_value_artifact "$ANSWER_VALUE_ARTIFACT" \
        --acec_artifact_v5 "$ACEC_ARTIFACT_V5" \
        --e5_model_path "$E5_MODEL_PATH" \
        --acec_nli_model "$NLI_MODEL" \
        2>&1 | tee "$OUTPUT_DIR.log"

echo "[v7] frozen inference results: $OUTPUT_DIR/results.json"
