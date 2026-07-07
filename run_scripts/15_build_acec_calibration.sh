#!/bin/bash
# ACEC Week-1 calibration + go/no-go gate (design doc Section 9).
# Run after 14_run_acec_distractor_pilot.sh has produced a fresh records.jsonl
# with 'supporting_facts'. Pure CPU/light-GPU replay of already-logged
# trajectories — no new LLM inference.
#
# 用法:
#   NLI_MODEL=cross-encoder/nli-deberta-v3-base bash 15_build_acec_calibration.sh
#   (omit NLI_MODEL to use the E5-cosine stand-in — sanity check only)

set -e

RECORDS=${RECORDS:-$HOME/logs/r3rag-qwen-distractor-acec-pilot/records.jsonl}
E5_MODEL_PATH=${E5_MODEL_PATH:-$HOME/models/e5-base-v2}
OUT_DIR=${OUT_DIR:-$HOME/logs/acec_calibration_pilot}

EXTRA_ARGS=""
if [ -n "${NLI_MODEL}" ]; then
    EXTRA_ARGS="--nli_model ${NLI_MODEL}"
fi

echo "[$(date)] ACEC calibration: records=${RECORDS}"

# needs: numpy (core) + sentence-transformers if NLI_MODEL is set (pip install -e ~/rag/src/belief/acec[nli])
$HOME/rag/.venv/bin/python $HOME/run_scripts/build_acec_calibration.py \
    --records ${RECORDS} \
    --e5_model_path ${E5_MODEL_PATH} \
    --out_dir ${OUT_DIR} \
    ${EXTRA_ARGS}
