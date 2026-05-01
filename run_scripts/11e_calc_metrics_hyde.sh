#!/bin/bash
# 计算 HyDE 的指标

SPLIT_MODEL=/root/models/Qwen2.5-7B-Instruct
MODEL_NAME="r3rag-qwen-real-hyde"
EXP_NAME="hyde"
LOG_DIR=/root/logs/${MODEL_NAME}

cd /root/rag/benchmark/R3-RAG

lsof /dev/nvidia* 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
echo "GPU cleared"

CUDA_VISIBLE_DEVICES=0 /root/rag/.venv/bin/python src/cal_metric.py \
    --model_path ${SPLIT_MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --exp_name ${EXP_NAME} \
    > ${LOG_DIR}/metrics.log 2>&1

echo "[$(date)] 指标计算完成"
cat ${LOG_DIR}/metrics.log | grep -E "em_processed|f1_processed|judge|avg_docs" | tail -20
echo ""
echo "results.json: ${LOG_DIR}/results.json"
