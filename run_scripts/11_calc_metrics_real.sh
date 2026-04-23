#!/bin/bash
# 计算真实检索实验的指标
# 用法:
#   BELIEF=1 bash 11_calc_metrics_real.sh   -> belief run
#   bash 11_calc_metrics_real.sh             -> baseline run

SPLIT_MODEL=/home/boyuz5/models/Qwen2.5-7B-Instruct

if [ "${BELIEF:-0}" = "1" ]; then
    MODEL_NAME="r3rag-qwen-real-full-belief"
    EXP_NAME="real_belief_early_stop_th0.70"
else
    MODEL_NAME="r3rag-qwen-real-full-baseline"
    EXP_NAME="real_baseline_docs10"
fi

LOG_DIR=/home/boyuz5/logs/${MODEL_NAME}

cd /home/boyuz5/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 /home/boyuz5/rag/.venv/bin/python src/cal_metric.py \
    --model_path ${SPLIT_MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --exp_name ${EXP_NAME} \
    > ${LOG_DIR}/metrics.log 2>&1

echo "[$(date)] 指标计算完成"
cat ${LOG_DIR}/metrics.log
echo ""
echo "results.json: ${LOG_DIR}/results.json"
echo "results.csv:  ${LOG_DIR}/results.csv"
