#!/bin/bash
# 启动 E5 检索服务
# 用法:
#   MINI=1 bash 02_start_retriever.sh   -> 使用 mini 索引 (smoke test)
#   bash 02_start_retriever.sh           -> 使用全量索引

set -e

REPO_ROOT="$HOME"
ENV_FILE="$(dirname "$0")/.env_retriever"

HOST=$(hostname -I | awk '{print $1}')
PORT=8001
MODEL="$REPO_ROOT/models/e5-base-v2"

if [ "${MINI:-0}" = "1" ]; then
    INDEX="$REPO_ROOT/data/indices/e5_Flat_mini/e5_Flat.index"
    CORPUS="$REPO_ROOT/data/flashrag_datasets/retrieval-corpus/wiki18_mini.jsonl"
    echo "[$(date)] 启动 MINI retriever 服务: http://${HOST}:${PORT}"
else
    INDEX="$REPO_ROOT/data/indices/e5_Flat/e5_Flat.index"
    CORPUS="$REPO_ROOT/data/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl"
    echo "[$(date)] 启动全量 retriever 服务 (17.3M 语料, float32 索引约 60GB, 实测 64,559,075,373 字节 — CPU RAM 有限时先用 MINI=1): http://${HOST}:${PORT}"
fi

echo "HOST=${HOST}" > "$ENV_FILE"
echo "SPLIT_HOST=${HOST}" >> "$ENV_FILE"
# RETRIEVE_URL: training scripts (20_train_grpo_rsf.sh etc.) source this file
# and only fall back to their hardcoded default (wrong port) if this isn't set.
echo "RETRIEVE_URL=http://${HOST}:${PORT}/search" >> "$ENV_FILE"

LOG="$REPO_ROOT/logs/retriever.log"
mkdir -p "$REPO_ROOT/logs"
echo "[$(date)] === retriever starting ===" >> "${LOG}"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
"$REPO_ROOT/rag/.venv/bin/python" \
    "$REPO_ROOT/rag/benchmark/retriever/src/retrive_server.py" \
    --host ${HOST} \
    --port ${PORT} \
    --model_path "$MODEL" \
    --index_path "$INDEX" \
    --corpus_path "$CORPUS" \
    2>&1 | tee -a "${LOG}"

echo "[$(date)] === retriever exited ===" >> "${LOG}"
