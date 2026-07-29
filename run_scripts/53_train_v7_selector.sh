#!/usr/bin/env bash
# ACEC v7: train the small state/set-conditioned selector on CPU or one GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT="${INPUT:?set INPUT to counterfactual_scored.jsonl}"
ANSWER_VALUE_ARTIFACT="${ANSWER_VALUE_ARTIFACT:?set ANSWER_VALUE_ARTIFACT}"
ANSWER_VALUE_CROSSFIT_BUNDLE="${ANSWER_VALUE_CROSSFIT_BUNDLE:-${ANSWER_VALUE_ARTIFACT%.json}.crossfit.json}"
OUTPUT="${OUTPUT:-$(dirname "$INPUT")/set_selector_v7.json}"
EPOCHS="${EPOCHS:-100}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
SEED="${SEED:-20260729}"
DISABLE_FEATURES="${DISABLE_FEATURES:-}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$INPUT" "$ANSWER_VALUE_ARTIFACT" "$ANSWER_VALUE_CROSSFIT_BUNDLE"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done

disable_args=()
if [[ -n "$DISABLE_FEATURES" ]]; then
    IFS=',' read -r -a disabled_features <<< "$DISABLE_FEATURES"
    for feature in "${disabled_features[@]}"; do
        disable_args+=(--disable_feature "$feature")
    done
fi

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" \
    "$REPO_ROOT/rag/train/train_set_selector_v7.py" \
        --input "$INPUT" \
        --answer_value_artifact "$ANSWER_VALUE_ARTIFACT" \
        --answer_value_crossfit_bundle "$ANSWER_VALUE_CROSSFIT_BUNDLE" \
        --output "$OUTPUT" \
        --epochs "$EPOCHS" \
        --hidden_size "$HIDDEN_SIZE" \
        --selector_temperature 0.7 \
        --seed "$SEED" \
        "${disable_args[@]}"

echo "[v7] selector artifact: $OUTPUT"
