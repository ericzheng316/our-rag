#!/usr/bin/env bash
# R3-style processed EM/F1 for the already-saved frozen/ACEC held-out answers.
# This is CPU/API-only: it does not need a retriever or GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-$PERSIST_ROOT/logs/acec_v5_heldout_dev256_7e1b03e}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v5_r3_processed_em_$(date +%Y%m%d_%H%M%S)}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.aiping.cn/api/v1}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-Qwen2.5-72B-Instruct}"
API_KEY_ENV="${API_KEY_ENV:-AIPING_API_KEY}"
WORKERS="${WORKERS:-8}"
RETRIES="${RETRIES:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED="${SEED:-20260722}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python runtime not executable: $PYTHON" >&2
    exit 2
fi
if [[ -z "${!API_KEY_ENV:-}" ]]; then
    echo "Required API credential variable is not set: $API_KEY_ENV" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT/results.json" ]]; then
    echo "Refusing to overwrite completed output: $OUTPUT_ROOT/results.json" >&2
    exit 2
fi

for required in \
    manifest.jsonl \
    predictions_frozen_r3.jsonl \
    predictions_step25.jsonl \
    predictions_step50.jsonl \
    predictions_step75.jsonl \
    predictions_step100.jsonl; do
    if [[ ! -f "$INPUT_ROOT/$required" ]]; then
        echo "Missing input: $INPUT_ROOT/$required" >&2
        exit 2
    fi
done

"$PYTHON" -c 'import openai; print("[preflight] openai=" + openai.__version__)'
mkdir -p "$OUTPUT_ROOT"

echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] input=$INPUT_ROOT output=$OUTPUT_ROOT"
echo "[preflight] model=$EXTRACTOR_MODEL workers=$WORKERS retries=$RETRIES"
echo "[preflight] API key is read from $API_KEY_ENV and will not be logged"

PYTHONPATH="$REPO_ROOT/rag/train" \
    "$PYTHON" "$REPO_ROOT/rag/train/tests/test_eval_r3_processed_answers_api.py"

PYTHONUNBUFFERED=1 "$PYTHON" "$REPO_ROOT/rag/train/eval_r3_processed_answers_api.py" \
    --manifest "$INPUT_ROOT/manifest.jsonl" \
    --prediction "frozen_r3=$INPUT_ROOT/predictions_frozen_r3.jsonl" \
    --prediction "step25=$INPUT_ROOT/predictions_step25.jsonl" \
    --prediction "step50=$INPUT_ROOT/predictions_step50.jsonl" \
    --prediction "step75=$INPUT_ROOT/predictions_step75.jsonl" \
    --prediction "step100=$INPUT_ROOT/predictions_step100.jsonl" \
    --output_dir "$OUTPUT_ROOT" \
    --base_url "$OPENAI_BASE_URL" \
    --api_key_env "$API_KEY_ENV" \
    --model "$EXTRACTOR_MODEL" \
    --workers "$WORKERS" \
    --retries "$RETRIES" \
    --provider_routing \
    --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_ROOT/eval.log"

echo "Processed-EM results: $OUTPUT_ROOT/results.json"
