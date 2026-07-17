#!/usr/bin/env bash
# Build the selected-evidence ACEC calibration-v4 artifact from logged
# distractor trajectories.  V1-v3 builders and artifacts remain untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$HOME/data}"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
RECORDS="${RECORDS:-$PERSIST_ROOT/logs/r3rag-qwen-distractor-acec-pilot/records.jsonl}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
OUT_DIR="${OUT_DIR:-$PERSIST_ROOT/logs/acec_calibration_v4}"
UNMATCHED_SLOT_POLICY="${UNMATCHED_SLOT_POLICY:-unknown}"
K_MODE="${K_MODE:-fixed}"
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

echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] records=$RECORDS"
echo "[preflight] output=$OUT_DIR unmatched_slot_policy=$UNMATCHED_SLOT_POLICY"
echo "[preflight] k_mode=$K_MODE fixed_k=$FIXED_K tau_new=$TAU_NEW"

PYTHONPATH="$REPO_ROOT/rag/src" \
    "$PYTHON" "$REPO_ROOT/rag/src/belief/acec/tests/test_calibration_v4.py"

"$PYTHON" "$REPO_ROOT/run_scripts/build_acec_calibration_v4.py" \
    --records "$RECORDS" \
    --e5_model_path "$E5_MODEL_PATH" \
    --nli_model "$NLI_MODEL" \
    --out_dir "$OUT_DIR" \
    --dataset hotpotqa \
    --k_mode "$K_MODE" \
    --fixed_k "$FIXED_K" \
    --tau_new "$TAU_NEW" \
    --unmatched_slot_policy "$UNMATCHED_SLOT_POLICY"

echo "ACEC-v4 artifact: $OUT_DIR/acec_calibration_v4.json"
echo "Label audit: $OUT_DIR/acec_calibration_v4_label_audit.jsonl"
