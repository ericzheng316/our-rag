#!/usr/bin/env bash
# Build the additive ACEC-v5 marginal-utility artifact. V1-v4 remain untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
RECORDS="${RECORDS:-$PERSIST_ROOT/logs/r3rag-qwen-distractor-acec-pilot/records.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-$PERSIST_ROOT/datasets/hotpotqa/distractor_jsonl/dev.jsonl}"
EVIDENCE_ADAPTER="${EVIDENCE_ADAPTER:-hotpotqa}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
OUT_DIR="${OUT_DIR:-$PERSIST_ROOT/logs/acec_calibration_v5}"
K_MODE="${K_MODE:-auto}"
FIXED_K="${FIXED_K:-2}"
TAU_NEW="${TAU_NEW:-0.90}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python runtime not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -f "$RECORDS" ]]; then
    echo "Calibration records not found: $RECORDS" >&2
    exit 2
fi
if [[ ! -f "$ANNOTATIONS" ]]; then
    echo "Evidence annotations not found: $ANNOTATIONS" >&2
    exit 2
fi

echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] records=$RECORDS"
echo "[preflight] annotations=$ANNOTATIONS adapter=$EVIDENCE_ADAPTER"
echo "[preflight] output=$OUT_DIR k_mode=$K_MODE fixed_k=$FIXED_K tau_new=$TAU_NEW"

PYTHONPATH="$REPO_ROOT/rag/src" \
    "$PYTHON" "$REPO_ROOT/rag/src/belief/acec/tests/test_calibration_v5.py"

"$PYTHON" "$REPO_ROOT/run_scripts/build_acec_calibration_v5.py" \
    --records "$RECORDS" \
    --annotations "$ANNOTATIONS" \
    --evidence_adapter "$EVIDENCE_ADAPTER" \
    --e5_model_path "$E5_MODEL_PATH" \
    --nli_model "$NLI_MODEL" \
    --out_dir "$OUT_DIR" \
    --k_mode "$K_MODE" \
    --fixed_k "$FIXED_K" \
    --tau_new "$TAU_NEW"

echo "ACEC-v5 artifact: $OUT_DIR/acec_calibration_v5.json"
echo "Marginal audit: $OUT_DIR/acec_calibration_v5_marginal_audit.jsonl"
