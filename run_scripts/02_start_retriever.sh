#!/bin/bash
# 启动 E5 检索服务
# 用法:
#   MINI=1 bash 02_start_retriever.sh   -> 使用 mini 索引 (smoke test)
#   bash 02_start_retriever.sh           -> 使用全量索引

set -e

HOST=$(hostname -I | awk '{print $1}')
PORT=8001
MODEL=/root/models/e5-base-v2

if [ "${MINI:-0}" = "1" ]; then
    INDEX=/root/data/indices/e5_Flat_mini/e5_Flat.index
    CORPUS=/root/data/flashrag_datasets/retrieval-corpus/wiki18_mini.jsonl
    echo "[$(date)] 启动 MINI retriever 服务: http://${HOST}:${PORT}"
else
    INDEX=/root/data/indices/e5_Flat/e5_Flat.index
    CORPUS=/root/data/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl
    echo "[$(date)] 启动全量 retriever 服务: http://${HOST}:${PORT}"
fi

echo "HOST=${HOST}" > /root/run_scripts/.env_retriever
echo "SPLIT_HOST=${HOST}" >> /root/run_scripts/.env_retriever

LOG=/root/logs/retriever.log
mkdir -p /root/logs
echo "[$(date)] === retriever starting ===" >> ${LOG}

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
conda run --no-capture-output -n rag python3 \
    /root/rag/benchmark/retriever/src/retrive_server.py \
    --host ${HOST} \
    --port ${PORT} \
    --model_path ${MODEL} \
    --index_path ${INDEX} \
    --corpus_path ${CORPUS} \
    2>&1 | tee -a ${LOG}

echo "[$(date)] === retriever exited ===" >> ${LOG}
