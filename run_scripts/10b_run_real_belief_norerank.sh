#!/bin/bash
# Ablation: belief + no cross-turn reranking
# 用法: bash 10b_run_real_belief_norerank.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-real-belief-norerank"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] Ablation: belief + no rerank: hotpotqa dev (7405条)"
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
    --belief_threshold 0.70 \
    --e5_model_path /root/models/e5-base-v2 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
