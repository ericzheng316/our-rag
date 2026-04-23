#!/bin/bash
# 全量 HotpotQA distractor baseline (7405条, docs=10, no belief)
# 用法: bash 06_run_full_baseline.sh

set -e

MODEL=/home/boyuz5/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/home/boyuz5/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-distractor-full-baseline"
LOG_DIR=/home/boyuz5/logs/${MODEL_NAME}
DISTRACTOR_FILE="${DATASET_ROOT}/hotpotqa/dev_distractor.jsonl"

mkdir -p ${LOG_DIR}
echo "[$(date)] 全量 distractor baseline: hotpotqa dev (7405条), docs=10, no belief"
echo "  log: ${LOG_DIR}/inference.log"

cd /home/boyuz5/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 DATASET_ROOT=${DATASET_ROOT} \
/home/boyuz5/rag/.venv/bin/python src/inference_new.py \
    --model_path ${MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id ${STOP_TOKEN_ID} \
    --num_of_docs 10 \
    --tp 1 \
    --datasets hotpotqa \
    --dev_file dev_distractor.jsonl \
    --distractor_file ${DISTRACTOR_FILE} \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
