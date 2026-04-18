#!/bin/bash
# 启动 E5 检索服务
# 用法:
#   MINI=1 bash 02_start_retriever.sh   -> 使用 mini 索引 (smoke test)
#   bash 02_start_retriever.sh           -> 使用全量索引

set -e

HOST=$(hostname -I | awk '{print $1}')
PORT=8001
MODEL=/home/boyuz5/models/e5-base-v2

if [ "${MINI:-0}" = "1" ]; then
    INDEX=/home/boyuz5/data/indices/e5_Flat_mini/e5_Flat.index
    CORPUS=/home/boyuz5/data/flashrag_datasets/retrieval-corpus/wiki18_mini.jsonl
    echo "[$(date)] 启动 MINI retriever 服务: http://${HOST}:${PORT}"
else
    INDEX=/home/boyuz5/data/indices/e5_Flat/e5_Flat.index
    CORPUS=/home/boyuz5/data/flashrag_datasets/retrieval-corpus/wiki18_100w_clean.jsonl
    echo "[$(date)] 启动全量 retriever 服务: http://${HOST}:${PORT}"
fi

echo "HOST=${HOST}" > /home/boyuz5/run_scripts/.env_retriever
echo "SPLIT_HOST=${HOST}" >> /home/boyuz5/run_scripts/.env_retriever

conda run -n rag python3 /home/boyuz5/rag/benchmark/retriever/src/retrive_server.py \
    --host ${HOST} \
    --port ${PORT} \
    --model_path ${MODEL} \
    --index_path ${INDEX} \
    --corpus_path ${CORPUS}
