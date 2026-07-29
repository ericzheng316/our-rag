#!/usr/bin/env bash
# Offline Gate-3 evaluation: identical q/answer/ordered trace, ACEC vs raw NLI.
# The script never calls the LLM or retriever and refuses mutable output paths.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$PERSIST_ROOT/rag-venv/bin/python}"
TRACE_PATH="${TRACE_PATH:-$PERSIST_ROOT/logs/acec_v63/fixed_trace.jsonl}"
GOLD_PATH="${GOLD_PATH:-$PERSIST_ROOT/logs/acec_v63/official_gold_subset.json}"
OFFICIAL_EVALUATOR="${OFFICIAL_EVALUATOR:-$PERSIST_ROOT/tools/hotpot_evaluate_v1.py}"
ACEC_ARTIFACT="${ACEC_ARTIFACT:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/logs/acec_v63/fixed_trace_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
SHARED_THRESHOLD="${SHARED_THRESHOLD:?set SHARED_THRESHOLD frozen before evaluation}"
THRESHOLD_PROVENANCE="${THRESHOLD_PROVENANCE:-frozen_unsupervised}"
NLI_MODEL_VERSION="${NLI_MODEL_VERSION:-cross-encoder/nli-deberta-v3-base}"
NLI_MODEL="${NLI_MODEL:-}"
NLI_GPU="${NLI_GPU:-}"
MAX_EVIDENCE="${MAX_EVIDENCE:-}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
SEED="${SEED:-20260723}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
  echo "[v6.3 preflight] refusing to run while .git/index.lock exists" >&2
  exit 2
fi
for path in "$PYTHON" "$TRACE_PATH" "$GOLD_PATH" "$OFFICIAL_EVALUATOR" "$ACEC_ARTIFACT"; do
  if [[ ! -s "$path" ]]; then
    echo "[v6.3 preflight] required file is missing/empty: $path" >&2
    exit 2
  fi
done
if ! "$PYTHON" -c 'import ujson' >/dev/null 2>&1; then
  echo "[v6.3 preflight] official HotpotQA evaluator requires ujson in $PYTHON" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[v6.3 preflight] refusing to overwrite: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -n "$NLI_MODEL" && -z "$NLI_GPU" ]]; then
  echo "[v6.3 preflight] set NLI_GPU when --nli_model must score missing candidates" >&2
  exit 2
fi
if [[ "$THRESHOLD_PROVENANCE" == "hotpot_threshold_supervised" ]]; then
  if [[ -z "${THRESHOLD_FIT_ID_HASH:-}" || -z "${THRESHOLD_VALIDATION_ID_HASH:-}" ]]; then
    echo "[v6.3 preflight] supervised threshold requires fit and validation id hashes" >&2
    exit 2
  fi
  if [[ "$THRESHOLD_FIT_ID_HASH" == "$THRESHOLD_VALIDATION_ID_HASH" ]]; then
    echo "[v6.3 preflight] threshold fit/validation id hashes must differ" >&2
    exit 2
  fi
fi

args=(
  --trace "$TRACE_PATH"
  --gold "$GOLD_PATH"
  --official_evaluator "$OFFICIAL_EVALUATOR"
  --acec_artifact "$ACEC_ARTIFACT"
  --output_dir "$OUTPUT_DIR"
  --shared_threshold "$SHARED_THRESHOLD"
  --threshold_provenance "$THRESHOLD_PROVENANCE"
  --nli_model_version "$NLI_MODEL_VERSION"
  --bootstrap_iterations "$BOOTSTRAP_ITERATIONS"
  --seed "$SEED"
)
if [[ -n "$NLI_MODEL" ]]; then
  args+=(--nli_model "$NLI_MODEL")
fi
if [[ -n "$MAX_EVIDENCE" ]]; then
  args+=(--max_evidence "$MAX_EVIDENCE")
fi
if [[ -n "${THRESHOLD_FIT_ID_HASH:-}" ]]; then
  args+=(--threshold_fit_id_hash "$THRESHOLD_FIT_ID_HASH")
fi
if [[ -n "${THRESHOLD_VALIDATION_ID_HASH:-}" ]]; then
  args+=(--threshold_validation_id_hash "$THRESHOLD_VALIDATION_ID_HASH")
fi

echo "[v6.3] fixed trace: $TRACE_PATH"
echo "[v6.3] immutable output: $OUTPUT_DIR"
echo "[v6.3] threshold: $SHARED_THRESHOLD ($THRESHOLD_PROVENANCE)"

export PYTHONPATH="$REPO_ROOT/rag/src:$REPO_ROOT/rag/train${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$NLI_GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$NLI_GPU"
fi
"$PYTHON" "$REPO_ROOT/rag/train/eval_hotpot_joint_v63.py" "${args[@]}"

test -s "$OUTPUT_DIR/results.json"
echo "[v6.3] COMPLETE: $OUTPUT_DIR/results.json"
