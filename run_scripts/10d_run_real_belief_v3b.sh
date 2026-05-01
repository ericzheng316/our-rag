#!/bin/bash
# BeliefState v3b: 三个 bug 修复后重跑
#   Fix 1: obs_extractor: if pool_docs: (not is not None) → turn 0 走 Z-score 路
#   Fix 2: budget 公式改为 round(exp(1.5*deficit))-1 → deficit≈0 时 extra=0
#   Fix 3: Condition A 阈值提高 0.85 → 0.92 减少误停
#   + --no_rerank: 禁用有害的 RRF 重排
# 用法: bash 10d_run_real_belief_v3b.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-real-belief-v3b"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] Belief v3b inference: hotpotqa dev (7405条)"
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
    --no_rerank \
    --e5_model_path /root/models/e5-base-v2 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
