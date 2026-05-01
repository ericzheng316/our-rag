#!/bin/bash
# BeliefState v3: Tasks 1-4 全部启用
#   Task 1: Z-score calibrated retrieval signal (top-100 background pool)
#   Task 2: Logit margin LLM reliability (logprobs=5)
#   Task 3: Exponential budget + dual-condition stopping (Cond-A: llm>0.85, Cond-B: ΔQ<0.05×2)
#   Task 4: RRF cross-turn reranking (k=60)
# 用法: bash 10c_run_real_belief_v3.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-real-belief-v3"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] Belief v3 inference: hotpotqa dev (7405条)"
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
    --e5_model_path /root/models/e5-base-v2 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
