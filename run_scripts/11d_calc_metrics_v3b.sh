#!/bin/bash
# 计算 belief v3b 的指标

SPLIT_MODEL=/root/models/Qwen2.5-7B-Instruct
MODEL_NAME="r3rag-qwen-real-belief-v3b"
EXP_NAME="belief_v3b"
LOG_DIR=/root/logs/${MODEL_NAME}

cd /root/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 /root/rag/.venv/bin/python src/cal_metric.py \
    --model_path ${SPLIT_MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --exp_name ${EXP_NAME} \
    > ${LOG_DIR}/metrics.log 2>&1

echo "[$(date)] 指标计算完成"
cat ${LOG_DIR}/metrics.log
echo ""
echo "results.json: ${LOG_DIR}/results.json"
