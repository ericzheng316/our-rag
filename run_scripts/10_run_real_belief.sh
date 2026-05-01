#!/bin/bash
# 真实检索 HotpotQA belief early-stopping (7405条)
# Belief 动作: confidence > threshold → 停止检索，强制输出答案
# 前提：02_start_retriever.sh 和 03_start_split_server.sh 已在后台启动
# 用法: bash 10_run_real_belief.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-real-full-belief"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] 真实检索 belief early-stopping: hotpotqa dev (7405条), threshold=0.70"
echo "  retriever: http://${HOST}:8001/search"
echo "  split:     http://${SPLIT_HOST}:8002/split_query"
echo "  log:       ${LOG_DIR}/inference.log"

cd /root/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 DATASET_ROOT=${DATASET_ROOT} \
/root/rag/.venv/bin/python src/inference_new.py \
    --model_path ${MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id ${STOP_TOKEN_ID} \
    --num_of_docs 10 \
    --tp 1 \
    --datasets hotpotqa \
    --dev_file dev.jsonl \
    --retrieve_url http://${HOST}:8001/search \
    --split_url http://${SPLIT_HOST}:8002/split_query \
    --use_belief \
    --belief_threshold 0.70 \
    --e5_model_path /root/models/e5-base-v2 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
