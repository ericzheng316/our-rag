#!/usr/bin/env bash
# Cached R3 processed-EM/F1 over the matched v6/v5/outcome held-out outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"

: "${INPUT_ROOT:?Set INPUT_ROOT to run_scripts/35 output directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/v6_v5_outcome_processed_em_$(date +%Y%m%d_%H%M%S)}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.aiping.cn/api/v1}"
AIPING_API_KEY="${AIPING_API_KEY:-}"
WORKERS="${WORKERS:-8}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ -z "$AIPING_API_KEY" ]]; then
    echo "AIPING_API_KEY must be exported in the shell" >&2
    exit 2
fi
for name in \
    frozen_r3 outcome_step50 outcome_step100 \
    v5_step50 v5_step100 v6_step50 v6_step100; do
    test -f "$INPUT_ROOT/predictions_${name}.jsonl"
done
test -f "$INPUT_ROOT/manifest.jsonl"
if [[ -e "$OUTPUT_ROOT/results.json" ]]; then
    echo "Refusing to overwrite completed evaluation: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[preflight] input=$INPUT_ROOT output=$OUTPUT_ROOT workers=$WORKERS"
echo "[preflight] extractor_cache_key=(question,prediction)"

AIPING_API_KEY="$AIPING_API_KEY" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$REPO_ROOT/rag/train/eval_r3_processed_answers_api_v6.py" \
        --manifest "$INPUT_ROOT/manifest.jsonl" \
        --prediction "frozen_r3=$INPUT_ROOT/predictions_frozen_r3.jsonl" \
        --prediction "outcome_step50=$INPUT_ROOT/predictions_outcome_step50.jsonl" \
        --prediction "outcome_step100=$INPUT_ROOT/predictions_outcome_step100.jsonl" \
        --prediction "v5_step50=$INPUT_ROOT/predictions_v5_step50.jsonl" \
        --prediction "v5_step100=$INPUT_ROOT/predictions_v5_step100.jsonl" \
        --prediction "v6_step50=$INPUT_ROOT/predictions_v6_step50.jsonl" \
        --prediction "v6_step100=$INPUT_ROOT/predictions_v6_step100.jsonl" \
        --output_dir "$OUTPUT_ROOT" \
        --base_url "$OPENAI_BASE_URL" \
        --model Qwen2.5-72B-Instruct \
        --workers "$WORKERS" \
        --provider_routing \
        2>&1 | tee "$OUTPUT_ROOT/eval.log"

echo "Processed-EM results: $OUTPUT_ROOT/results.json"
