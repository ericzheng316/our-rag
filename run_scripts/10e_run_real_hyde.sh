#!/bin/bash
# HyDE (Hypothetical Document Embeddings) for second hop
#   Turn 1: normal sub-query encoding
#   Turn 2+: generate hypothetical Wikipedia passage via split server,
#            encode passage instead of query → closes query-doc embedding gap
#   Hypothesis: +3-5pt EM on bridge questions (80% of HotpotQA)
# 用法: bash 10e_run_real_hyde.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="r3rag-qwen-real-hyde"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] HyDE inference: hotpotqa dev (7405条)"
echo "  retriever: http://${HOST}:8001/search"
echo "  split:     http://${SPLIT_HOST}:8002/split_query"
echo "  hyde:      http://${SPLIT_HOST}:8002/hyde_passage"
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
    --use_hyde \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
