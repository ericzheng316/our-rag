#!/usr/bin/env bash
# ACEC v7 G3: verify that conditional delta-C adds value beyond the reranker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
FULL_EVAL="${FULL_EVAL:?set FULL_EVAL to full selector slates_scored.jsonl}"
NO_DELTA_C_EVAL="${NO_DELTA_C_EVAL:?set NO_DELTA_C_EVAL to ablated slates_scored.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "$FULL_EVAL")/delta_c_ablation.json}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$FULL_EVAL" "$NO_DELTA_C_EVAL"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done

"$PYTHON" "$REPO_ROOT/rag/train/compare_selector_delta_c_ablation_v7.py" \
    --full "$FULL_EVAL" \
    --no_delta_c "$NO_DELTA_C_EVAL" \
    --output "$OUTPUT" \
    --bootstrap_iterations 10000

echo "[v7] delta-C ablation gate: $OUTPUT"
