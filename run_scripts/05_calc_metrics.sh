#!/bin/bash
# 计算 EM/F1 指标，对应论文 Table 1
# 用法:
#   MINI=1 bash 05_calc_metrics.sh   -> mini smoke test 结果
#   bash 05_calc_metrics.sh           -> 全量结果

if [ "${DISTRACTOR:-0}" = "1" ]; then
    MODEL_NAME="r3rag-qwen-distractor-mini"
elif [ "${MINI:-0}" = "1" ]; then
    MODEL_NAME="r3rag-qwen-e5-mini"
else
    MODEL_NAME="r3rag-qwen-e5"
fi
LOG_DIR=/home/boyuz5/logs/${MODEL_NAME}
SPLIT_MODEL=/home/boyuz5/models/Qwen2.5-7B-Instruct

cd /home/boyuz5/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 /home/boyuz5/rag/.venv/bin/python src/cal_metric.py \
    --model_path ${SPLIT_MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    > ${LOG_DIR}/metrics.log 2>&1

echo "[$(date)] 指标计算完成，结果在 ${LOG_DIR}/metrics.log"
cat ${LOG_DIR}/metrics.log
