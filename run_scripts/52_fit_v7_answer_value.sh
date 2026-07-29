#!/usr/bin/env bash
# ACEC v7: fit the question-disjoint frozen signed answer-value head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT="${INPUT:?set INPUT to counterfactual_scored.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "$INPUT")/answer_value_v7.json}"
CROSSFIT_OUTPUT="${CROSSFIT_OUTPUT:-$(dirname "$INPUT")/answer_value_v7.crossfit.json}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" \
    "$REPO_ROOT/rag/train/fit_answer_value_v7.py" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --crossfit_output "$CROSSFIT_OUTPUT" \
        --fit_split fit \
        --validation_split validation \
        --folds 5 \
        --l2 1.0

echo "[v7] answer-value artifact: $OUTPUT"
echo "[v7] answer-value cross-fit bundle: $CROSSFIT_OUTPUT"
