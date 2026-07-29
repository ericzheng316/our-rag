#!/usr/bin/env bash
# ACEC v7: add the frozen learned slate, then score it with the same answerer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT="${INPUT:?set INPUT to held-out counterfactual_slates.jsonl}"
SELECTOR_ARTIFACT="${SELECTOR_ARTIFACT:?set SELECTOR_ARTIFACT}"
ANSWER_VALUE_ARTIFACT="${ANSWER_VALUE_ARTIFACT:?set ANSWER_VALUE_ARTIFACT}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT")/eval_v7}"
K="${K:-5}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
ORACLE_INPUT="${ORACLE_INPUT:-$(dirname "$INPUT")/oracle_headroom.jsonl}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$INPUT" "$SELECTOR_ARTIFACT" "$ANSWER_VALUE_ARTIFACT" "$MODEL_PATH" "$ORACLE_INPUT"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[v7] refusing to overwrite: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" \
    "$REPO_ROOT/rag/train/eval_set_selector_v7.py" \
        --input "$INPUT" \
        --selector_artifact "$SELECTOR_ARTIFACT" \
        --answer_value_artifact "$ANSWER_VALUE_ARTIFACT" \
        --output "$OUTPUT_DIR/slates_with_learned.jsonl" \
        --selected_k "$K" \
        --split "$EVAL_SPLIT"

"$PYTHON" "$REPO_ROOT/rag/train/score_counterfactual_answer_value_v7.py" \
    --input "$OUTPUT_DIR/slates_with_learned.jsonl" \
    --output "$OUTPUT_DIR/slates_scored.jsonl" \
    --model "$MODEL_PATH" \
    --device auto \
    --dtype bfloat16

"$PYTHON" "$REPO_ROOT/rag/train/summarize_selector_eval_v7.py" \
    --input "$OUTPUT_DIR/slates_scored.jsonl" \
    --output "$OUTPUT_DIR/selector_heldout_eval.json" \
    --bootstrap_iterations 10000

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" \
    "$REPO_ROOT/rag/train/eval_selected_calibration_v7.py" \
        --selection "$OUTPUT_DIR/slates_with_learned.jsonl" \
        --oracle "$ORACLE_INPUT" \
        --output "$OUTPUT_DIR/selected_calibration.json"

echo "[v7] frozen selector evaluation: $OUTPUT_DIR/slates_scored.jsonl"
echo "[v7] held-out gate: $OUTPUT_DIR/selector_heldout_eval.json"
