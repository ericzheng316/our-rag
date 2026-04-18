#!/bin/bash
# 启动 split query 服务（用本地 Qwen2.5-7B 代替论文中的 72B）
# 注意：7B 的 query 改写质量低于 72B，结果会略低于论文数值

set -e

HOST=$(hostname -I | awk '{print $1}')
PORT=8002
MODEL=/home/boyuz5/models/Qwen2.5-7B-Instruct

echo "[$(date)] 启动 split query 服务（Qwen2.5-7B），地址: http://${HOST}:${PORT}"
echo "SPLIT_HOST=${HOST}" >> /home/boyuz5/run_scripts/.env_retriever

cd /home/boyuz5/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 /home/boyuz5/rag/.venv/bin/python src/split_server.py \
    --host ${HOST} \
    --port ${PORT} \
    --model_path ${MODEL} \
    --tp 1
