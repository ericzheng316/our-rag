#!/bin/bash
# 启动 E5 检索服务
# 用法:
#   MINI=1 bash 02_start_retriever.sh   -> 使用 mini 索引 (smoke test)
#   bash 02_start_retriever.sh           -> 使用全量索引
#   GPU_ID=1 FAISS_GPU=1 bash 02_start_retriever.sh
#       -> 双卡场景:检索(E5 编码 + FAISS 搜索)整体挪到第二张卡,
#          训练那张卡(GPU 0)不再跟检索抢显存/算力。FAISS_GPU=1 把
#          faiss_gpu 从 False 打开 —— FlashRAG 的 DenseRetriever 用
#          faiss.index_cpu_to_all_gpus(...)(retriever.py:399),即"把索引
#          搬到这个进程能看到的所有 GPU 上",所以只要 GPU_ID 把
#          CUDA_VISIBLE_DEVICES 锁到第二张卡,索引就只会搬到那一张卡,
#          不会跟训练那张卡冲突。默认(不设这两个变量)保持现在单卡 CPU-
#          FAISS 的行为不变,没拿到第二张卡之前不影响任何现有脚本。

set -e

GPU_ID="${GPU_ID:-0}"
FAISS_GPU="${FAISS_GPU:-0}"

REPO_ROOT="$HOME"
ENV_FILE="$(dirname "$0")/.env_retriever"

# 默认保持原行为（绑本机 LAN 地址、8001），但允许覆盖：共享机器上应绑回环并
# 换非默认端口，既避免与他人的服务撞端口，也不把检索服务暴露给整个集群。
HOST="${RETRIEVER_HOST:-$(hostname -I | awk '{print $1}')}"
PORT="${RETRIEVER_PORT:-8001}"
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

echo "[$(date)] GPU_ID=${GPU_ID} FAISS_GPU=${FAISS_GPU}"

CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
FAISS_GPU=${FAISS_GPU} \
"${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}" \
    "$REPO_ROOT/rag/benchmark/retriever/src/retrive_server.py" \
    --host ${HOST} \
    --port ${PORT} \
    --model_path "$MODEL" \
    --index_path "$INDEX" \
    --corpus_path "$CORPUS" \
    2>&1 | tee -a "${LOG}"

echo "[$(date)] === retriever exited ===" >> "${LOG}"
