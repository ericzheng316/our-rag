#!/bin/bash
# NIM API baseline: meta/llama-3.1-8b-instruct (hotpotqa dev 7405条)
# 不需要本地 GPU，也不需要 vllm 加载模型。
# 前提：02_start_retriever.sh 和 03_start_split_server.sh 已启动（用于真实检索）。
# 用法: bash 12_run_nim_llama31.sh

set -e

source /root/run_scripts/.env_retriever

NIM_KEYS_FILE=/root/rag/nim_keys.json
NIM_MODEL="meta/llama-3.1-8b-instruct"
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="nim-llama31-8b-real-baseline"
LOG_DIR=/root/logs/${MODEL_NAME}

mkdir -p ${LOG_DIR}
echo "[$(date)] NIM baseline: ${NIM_MODEL}, hotpotqa dev (7405条), docs=10"
echo "  retriever: http://${HOST}:8001/search"
echo "  split:     http://${SPLIT_HOST}:8002/split_query"
echo "  log:       ${LOG_DIR}/inference.log"

cd /root/rag/benchmark/R3-RAG

DATASET_ROOT=${DATASET_ROOT} \
/root/rag/.venv/bin/python src/inference_new.py \
    --model_path dummy \
    --nim_keys_file ${NIM_KEYS_FILE} \
    --nim_model ${NIM_MODEL} \
    --nim_workers 21 \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id 128009 \
    --num_of_docs 10 \
    --datasets hotpotqa \
    --dev_file dev.jsonl \
    --retrieve_url http://${HOST}:8001/search \
    --split_url http://${SPLIT_HOST}:8002/split_query \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
