#!/bin/bash
# IRCoT baseline on HotpotQA real retrieval (7405条)
# 前提：02_start_retriever.sh 已启动 (retriever on :8001)
# 用法: bash 12_run_ircot_hotpotqa.sh

set -e

source /root/run_scripts/.env_retriever

MODEL=/root/models/Qwen2.5-7B-Instruct
DATASET_ROOT=/root/data/flashrag_datasets
MODEL_NAME="ircot-qwen7b-real"
LOG_DIR=/root/logs/${MODEL_NAME}
OUTPUT_FILE=${LOG_DIR}/records.jsonl

mkdir -p ${LOG_DIR}
echo "[$(date)] IRCoT baseline: hotpotqa dev (7405条), topk=5, max_turns=6"
echo "  model:     ${MODEL}"
echo "  retriever: http://${HOST}:8001/search"
echo "  output:    ${OUTPUT_FILE}"
echo "  log:       ${LOG_DIR}/inference.log"

CUDA_VISIBLE_DEVICES=0 /root/rag/.venv/bin/python \
    /root/run_scripts/ircot_inference.py \
    --model_path   ${MODEL} \
    --retrieve_url http://${HOST}:8001/search \
    --dev_file     ${DATASET_ROOT}/hotpotqa/dev.jsonl \
    --output_file  ${OUTPUT_FILE} \
    --topk         5 \
    --max_turns    6 \
    --max_new_tokens 150 \
    --tp           1 \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成，开始计算 metrics"
/root/rag/.venv/bin/python /root/run_scripts/ircot_eval.py \
    --records_file ${OUTPUT_FILE} \
    --output_dir   ${LOG_DIR}

echo "[$(date)] 全部完成"
