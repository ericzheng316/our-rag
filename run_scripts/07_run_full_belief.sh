#!/bin/bash
# 全量 HotpotQA distractor belief run (7405条, docs=10 adaptive, use_belief)
# Belief 动作: 根据 Beta-Bernoulli 置信度动态调整每步检索文档数
#   confidence >= 0.70 → 取 top-5 docs (减少噪声)
#   confidence <  0.70 → 取 top-10 docs (继续探索)
# 用法: bash 07_run_full_belief.sh

set -e

MODEL=/home/boyuz5/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/home/boyuz5/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-distractor-full-belief"
LOG_DIR=/home/boyuz5/logs/${MODEL_NAME}
DISTRACTOR_FILE="${DATASET_ROOT}/hotpotqa/dev_distractor.jsonl"

mkdir -p ${LOG_DIR}
echo "[$(date)] 全量 distractor belief: hotpotqa dev (7405条), docs=10 adaptive, belief_threshold=0.70"
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
    --use_belief \
    --belief_threshold 0.70 \
    --e5_model_path /home/boyuz5/models/e5-base-v2 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
