#!/bin/bash
# 运行 R3-RAG 推理
# 用法:
#   DISTRACTOR=1 bash 04_run_inference.sh  -> distractor mini (200样本, 无需任何服务)
#   MINI=1 bash 04_run_inference.sh        -> smoke test (hotpotqa 200样本, mini retriever)
#   bash 04_run_inference.sh               -> 全量评测 (3个数据集)
# DISTRACTOR=1 模式不需要启动 retriever/split server

set -e

MODEL=/home/boyuz5/models/R3-RAG-Qwen
STOP_TOKEN_ID=151645
DATASET_ROOT=/home/boyuz5/data/flashrag_datasets

if [ "${DISTRACTOR:-0}" = "1" ]; then
    MODEL_NAME="r3rag-qwen-distractor-mini"
    DATASETS="hotpotqa"
    DEV_FILE="dev_mini_distractor.jsonl"
    DISTRACTOR_FILE="${DATASET_ROOT}/hotpotqa/dev_mini_distractor.jsonl"
    NUM_DOCS=10
    echo "[$(date)] DISTRACTOR 模式: hotpotqa/dev_mini_distractor.jsonl (200样本, 无需检索服务)"
    EXTRA_ARGS="--distractor_file ${DISTRACTOR_FILE}"
elif [ "${MINI:-0}" = "1" ]; then
    source /home/boyuz5/run_scripts/.env_retriever
    MODEL_NAME="r3rag-qwen-e5-mini"
    DATASETS="hotpotqa"
    DEV_FILE="dev_mini.jsonl"
    NUM_DOCS=3
    EXTRA_ARGS="--retrieve_url http://${HOST}:8001/search --split_url http://${SPLIT_HOST}:8002/split_query"
    echo "[$(date)] MINI 模式: hotpotqa/dev_mini.jsonl (200样本)"
else
    source /home/boyuz5/run_scripts/.env_retriever
    MODEL_NAME="r3rag-qwen-e5"
    DATASETS="2wikimultihopqa,hotpotqa,musique"
    DEV_FILE="dev.jsonl"
    NUM_DOCS=3
    EXTRA_ARGS="--retrieve_url http://${HOST}:8001/search --split_url http://${SPLIT_HOST}:8002/split_query"
    echo "[$(date)] 全量模式: 3个数据集"
fi

LOG_DIR=/home/boyuz5/logs/${MODEL_NAME}
mkdir -p ${LOG_DIR}

echo "  model: ${MODEL}"
echo "  log:   ${LOG_DIR}/inference.log"

cd /home/boyuz5/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 DATASET_ROOT=${DATASET_ROOT} \
/home/boyuz5/rag/.venv/bin/python src/inference_new.py \
    --model_path ${MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id ${STOP_TOKEN_ID} \
    --num_of_docs ${NUM_DOCS} \
    --tp 1 \
    --datasets ${DATASETS} \
    --dev_file ${DEV_FILE} \
    ${EXTRA_ARGS} \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成"
