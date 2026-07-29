#!/usr/bin/env bash
# ACEC v7: frozen teacher-forced gold-answer utility for slate prefixes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT="${INPUT:?set INPUT to counterfactual_slates.jsonl}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
OUTPUT="${OUTPUT:-$(dirname "$INPUT")/counterfactual_scored.jsonl}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$INPUT" "$MODEL_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done

PYTHONUNBUFFERED=1 "$PYTHON" \
    "$REPO_ROOT/rag/train/score_counterfactual_answer_value_v7.py" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$MODEL_PATH" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --batch_size "$BATCH_SIZE"

ORACLE_INPUT="$(dirname "$INPUT")/oracle_headroom.jsonl"
if [[ -s "$ORACLE_INPUT" ]]; then
    "$PYTHON" "$REPO_ROOT/rag/train/eval_candidate_headroom_v7.py" \
        --input "$ORACLE_INPUT" \
        --answer_input "$OUTPUT" \
        --k "${K:-5}" \
        --pool_size "${L:-20}" \
        --output "$(dirname "$OUTPUT")/g1_headroom_complete.json"
fi

echo "[v7] scored answer-value data: $OUTPUT"
