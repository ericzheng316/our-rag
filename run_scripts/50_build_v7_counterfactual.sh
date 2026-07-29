#!/usr/bin/env bash
# ACEC v7 G0/G1: replay a frozen trace, build wide pools and counterfactual slates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"

TRAJECTORY="${TRAJECTORY:?set TRAJECTORY to a frozen v6 trajectory JSONL}"
DATASET="${DATASET:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/train_sf.jsonl}"
ACEC_ARTIFACT_V5="${ACEC_ARTIFACT_V5:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/logs/acec_v7_counterfactual_$(date -u +%Y%m%dT%H%M%SZ)}"
L="${L:-20}"
K="${K:-5}"
MAX_CONTEXT_DOCS="${MAX_CONTEXT_DOCS:-10}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-512}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v7] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
for path in "$PYTHON" "$TRAJECTORY" "$DATASET" "$ACEC_ARTIFACT_V5" "$E5_MODEL_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "[v7] missing required path: $path" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[v7] refusing to overwrite output: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$REPO_ROOT/rag/src:$REPO_ROOT/rag/train" \
    "$PYTHON" "$REPO_ROOT/rag/train/collect_candidate_pools_v7.py" \
        --trajectory "$TRAJECTORY" \
        --dataset "$DATASET" \
        --output "$OUTPUT_DIR/candidate_pools.jsonl" \
        --oracle_output "$OUTPUT_DIR/oracle_headroom.jsonl" \
        --retrieve_url "$RETRIEVE_URL" \
        --candidate_pool_size "$L" \
        --selected_k "$K" \
        --max_context_documents "$MAX_CONTEXT_DOCS" \
        --max_trajectories "$MAX_TRAJECTORIES" \
        --acec_artifact_v5 "$ACEC_ARTIFACT_V5" \
        --e5_model_path "$E5_MODEL_PATH" \
        --acec_nli_model "$NLI_MODEL"

PYTHONPATH="$REPO_ROOT/rag/src" \
    "$PYTHON" "$REPO_ROOT/rag/train/assign_question_splits_v7.py" \
        --input "$OUTPUT_DIR/candidate_pools.jsonl" \
        --output "$OUTPUT_DIR/candidate_pools_split.jsonl"

PYTHONPATH="$REPO_ROOT/rag/src" \
    "$PYTHON" "$REPO_ROOT/rag/train/build_counterfactual_sets_v7.py" \
        --input "$OUTPUT_DIR/candidate_pools_split.jsonl" \
        --output "$OUTPUT_DIR/counterfactual_slates.jsonl" \
        --candidate_pool_size "$L" \
        --selected_k "$K"

"$PYTHON" "$REPO_ROOT/rag/train/eval_candidate_headroom_v7.py" \
    --input "$OUTPUT_DIR/oracle_headroom.jsonl" \
    --k "$K" \
    --pool_size "$L" \
    --output "$OUTPUT_DIR/g1_headroom_recall_only.json"

echo "[v7] candidate/slate output: $OUTPUT_DIR"
