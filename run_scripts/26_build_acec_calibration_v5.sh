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
MIN_VALIDATION_EXAMPLES="${MIN_VALIDATION_EXAMPLES:-50}"
MIN_VALIDATION_FIT_ELIGIBLE_FRACTION="${MIN_VALIDATION_FIT_ELIGIBLE_FRACTION:-0.15}"
MIN_VALIDATION_POSTERIOR_GAIN_OVER_NOVELTY="${MIN_VALIDATION_POSTERIOR_GAIN_OVER_NOVELTY:-0.03}"
MIN_VALIDATION_NONREPEAT_EXAMPLES="${MIN_VALIDATION_NONREPEAT_EXAMPLES:-20}"
MIN_VALIDATION_NONREPEAT_RAW_SUPPORT_AUC="${MIN_VALIDATION_NONREPEAT_RAW_SUPPORT_AUC:-0.70}"
MIN_FIT_TARGET_EXAMPLES_PER_ACTION="${MIN_FIT_TARGET_EXAMPLES_PER_ACTION:-10}"
REQUIRED_ACTION_MODES="${REQUIRED_ACTION_MODES:-EXPAND,REWRITE,DECOMPOSE}"
MIN_DATASET_QUESTION_ID_FRACTION="${MIN_DATASET_QUESTION_ID_FRACTION:-1.0}"
MIN_CORPUS_DOCUMENT_ID_FRACTION="${MIN_CORPUS_DOCUMENT_ID_FRACTION:-0.95}"

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
echo "[preflight] validity=min_val=$MIN_VALIDATION_EXAMPLES min_eligible=$MIN_VALIDATION_FIT_ELIGIBLE_FRACTION"
echo "[preflight] min_gain_over_novelty=$MIN_VALIDATION_POSTERIOR_GAIN_OVER_NOVELTY"
echo "[preflight] actions=$REQUIRED_ACTION_MODES min_per_action=$MIN_FIT_TARGET_EXAMPLES_PER_ACTION"

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
    --tau_new "$TAU_NEW" \
    --min_validation_examples "$MIN_VALIDATION_EXAMPLES" \
    --min_validation_fit_eligible_fraction "$MIN_VALIDATION_FIT_ELIGIBLE_FRACTION" \
    --min_validation_posterior_gain_over_novelty "$MIN_VALIDATION_POSTERIOR_GAIN_OVER_NOVELTY" \
    --min_validation_nonrepeat_examples "$MIN_VALIDATION_NONREPEAT_EXAMPLES" \
    --min_validation_nonrepeat_raw_support_auc "$MIN_VALIDATION_NONREPEAT_RAW_SUPPORT_AUC" \
    --min_fit_target_examples_per_action "$MIN_FIT_TARGET_EXAMPLES_PER_ACTION" \
    --required_action_modes "$REQUIRED_ACTION_MODES" \
    --min_dataset_question_id_fraction "$MIN_DATASET_QUESTION_ID_FRACTION" \
    --min_corpus_document_id_fraction "$MIN_CORPUS_DOCUMENT_ID_FRACTION"

echo "ACEC-v5 artifact: $OUT_DIR/acec_calibration_v5.json"
echo "Marginal audit: $OUT_DIR/acec_calibration_v5_marginal_audit.jsonl"
