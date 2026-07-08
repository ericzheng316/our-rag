#!/bin/bash
# Week-2 D8-10 probe (design doc Section 9): live rollout with ACEC's VOI
# stopping gate + posterior-driven bridge-entity query rewriting turned on.
# Zero training — the cheap first check of how much of the design doc's
# diagnosed +27.9pt oracle bridge-injection headroom the belief recovers on
# its own.
#
# Same distractor-mode / NUM_DOCS rationale as 14_run_acec_distractor_pilot.sh
# (this box's 32GB CPU RAM can't hold the real-retrieval Flat index regardless
# of GPU size; NUM_DOCS<10 keeps hit/miss genuinely query-dependent).
#
# Needs OBSERVATION_MODEL pointed at a Week-1 observation_model.json (from
# run_scripts/15_build_acec_calibration.sh) — omitting it falls back to
# uncalibrated ACECConfig() defaults with a printed warning, which defeats
# the point of this probe (Section 2.2: test time should use frozen,
# calibrated parameters).
#
# For the "without ACEC" comparison arm, reuse 14_run_acec_distractor_pilot.sh
# with the same MAX_SAMPLES — no separate script needed.
#
# 用法:
#   MAX_SAMPLES=1000 NUM_DOCS=3 \
#   OBSERVATION_MODEL=$HOME/logs/acec_calibration_pilot/observation_model.json \
#   bash 16_run_acec_live_probe.sh

set -e

MODEL=${MODEL:-$HOME/models/R3-RAG-Qwen}
STOP_TOKEN_ID=151645
DATASET_ROOT=${DATASET_ROOT:-$HOME/data/flashrag_datasets}
MODEL_NAME="r3rag-qwen-distractor-acec-live-probe"
LOG_DIR=$HOME/logs/${MODEL_NAME}
DISTRACTOR_FILE="${DATASET_ROOT}/hotpotqa/dev_distractor.jsonl"
MAX_SAMPLES=${MAX_SAMPLES:-1000}
NUM_DOCS=${NUM_DOCS:-3}
E5_MODEL_PATH=${E5_MODEL_PATH:-$HOME/models/e5-base-v2}
NLI_MODEL=${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}
OBSERVATION_MODEL=${OBSERVATION_MODEL:-}

mkdir -p ${LOG_DIR}
echo "[$(date)] ACEC live probe: hotpotqa distractor, ${MAX_SAMPLES} samples, docs=${NUM_DOCS}, VOI gate + bridge rewriting ON"
if [ -z "${OBSERVATION_MODEL}" ]; then
    echo "  WARNING: OBSERVATION_MODEL not set — using uncalibrated ACECConfig() defaults"
else
    echo "  observation model: ${OBSERVATION_MODEL}"
fi
echo "  log: ${LOG_DIR}/inference.log"

cd $HOME/rag/benchmark/R3-RAG

OBS_MODEL_ARGS=""
if [ -n "${OBSERVATION_MODEL}" ]; then
    OBS_MODEL_ARGS="--acec_observation_model ${OBSERVATION_MODEL}"
fi

CUDA_VISIBLE_DEVICES=0 DATASET_ROOT=${DATASET_ROOT} \
$HOME/rag/.venv/bin/python src/inference_new.py \
    --model_path ${MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id ${STOP_TOKEN_ID} \
    --num_of_docs ${NUM_DOCS} \
    --tp 1 \
    --datasets hotpotqa \
    --dev_file dev_distractor.jsonl \
    --distractor_file ${DISTRACTOR_FILE} \
    --max_samples ${MAX_SAMPLES} \
    --use_acec \
    --e5_model_path ${E5_MODEL_PATH} \
    --acec_nli_model ${NLI_MODEL} \
    ${OBS_MODEL_ARGS} \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成 -> ${LOG_DIR}/records.jsonl"
echo "[$(date)] 对比 baseline: MAX_SAMPLES=${MAX_SAMPLES} bash 14_run_acec_distractor_pilot.sh (NUM_DOCS 记得保持一致)"
