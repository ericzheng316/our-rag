#!/bin/bash
# 启动 split query 服务（用本地 Qwen2.5-7B 代替论文中的 72B）
# 注意：7B 的 query 改写质量低于 72B，结果会略低于论文数值

set -e

REPO_ROOT="$HOME"
ENV_FILE="$(dirname "$0")/.env_retriever"

HOST=$(hostname -I | awk '{print $1}')
PORT=8002
MODEL="$REPO_ROOT/models/Qwen2.5-7B-Instruct"

echo "[$(date)] 启动 split query 服务（Qwen2.5-7B），地址: http://${HOST}:${PORT}"
echo "SPLIT_HOST=${HOST}" >> "$ENV_FILE"
# SPLIT_URL: training scripts source this file and only fall back to their
# hardcoded default (wrong port) if this isn't set.
echo "SPLIT_URL=http://${HOST}:${PORT}/split_query" >> "$ENV_FILE"

cd "$REPO_ROOT/rag/benchmark/R3-RAG"

CUDA_VISIBLE_DEVICES=0 "$REPO_ROOT/rag/.venv/bin/python" src/split_server.py \
    --host ${HOST} \
    --port ${PORT} \
    --model_path "$MODEL" \
    --tp 1
