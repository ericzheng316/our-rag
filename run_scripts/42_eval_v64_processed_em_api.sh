#!/usr/bin/env bash
# CPU/API-only R3 processed EM/F1 for selected v6.4 factorial arms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
INPUT_ROOT="${INPUT_ROOT:?set INPUT_ROOT to a completed v6.4 inference directory}"
PREDICTIONS="${PREDICTIONS:?set comma-separated NAME=/absolute/predictions.jsonl entries}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST_ROOT/logs/acec_v64_processed_$(date -u +%Y%m%dT%H%M%SZ)}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.aiping.cn/api/v1}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-Qwen2.5-72B-Instruct}"
API_KEY_ENV="${API_KEY_ENV:-AIPING_API_KEY}"
WORKERS="${WORKERS:-8}"
RETRIES="${RETRIES:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED="${SEED:-20260723}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v6.4 processed] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "[v6.4 processed] Python runtime is not executable: $PYTHON" >&2
    exit 2
fi
if [[ -z "${!API_KEY_ENV:-}" ]]; then
    echo "[v6.4 processed] required credential variable is unset: $API_KEY_ENV" >&2
    exit 2
fi
if [[ ! -s "$INPUT_ROOT/manifest.jsonl" ]]; then
    echo "[v6.4 processed] missing inference manifest: $INPUT_ROOT/manifest.jsonl" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT/results.json" ]]; then
    echo "[v6.4 processed] refusing to overwrite: $OUTPUT_ROOT/results.json" >&2
    exit 2
fi

prediction_args=()
IFS=',' read -r -a prediction_values <<< "$PREDICTIONS"
for value in "${prediction_values[@]}"; do
    if [[ "$value" != *=* ]]; then
        echo "[v6.4 processed] prediction must be NAME=/absolute/path: $value" >&2
        exit 2
    fi
    name="${value%%=*}"
    path="${value#*=}"
    if [[ -z "$name" || ! -s "$path" ]]; then
        echo "[v6.4 processed] prediction is missing/empty: $value" >&2
        exit 2
    fi
    prediction_args+=(--prediction "$name=$path")
done

"$PYTHON" -c 'import openai; print("[v6.4 processed] openai=" + openai.__version__)'
mkdir -p "$OUTPUT_ROOT"
echo "[v6.4 processed] variants=${#prediction_values[@]} output=$OUTPUT_ROOT"
echo "[v6.4 processed] API key is read from $API_KEY_ENV and will not be logged"

PYTHONPATH="$REPO_ROOT/rag/train" \
    "$PYTHON" "$REPO_ROOT/rag/train/tests/test_eval_r3_processed_answers_api.py"

PYTHONUNBUFFERED=1 "$PYTHON" \
    "$REPO_ROOT/rag/train/eval_r3_processed_answers_api_v6.py" \
    --manifest "$INPUT_ROOT/manifest.jsonl" \
    "${prediction_args[@]}" \
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

test -s "$OUTPUT_ROOT/results.json"
echo "[v6.4 processed] COMPLETE: $OUTPUT_ROOT/results.json"
