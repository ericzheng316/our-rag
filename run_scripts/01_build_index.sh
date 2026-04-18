#!/bin/bash
# 构建 E5 FAISS 索引
# 用法:
#   MINI=1 bash 01_build_index.sh   -> 100K 语料，快速测试用
#   bash 01_build_index.sh           -> 全量 17.3M 语料

set -e

CORPUS_DIR=/home/boyuz5/data/flashrag_datasets/retrieval-corpus
MODEL=/home/boyuz5/models/e5-base-v2
INDEX_BUILDER=/home/boyuz5/rag/tool/FlashRAG/flashrag/retriever/index_builder.py

if [ "${MINI:-0}" = "1" ]; then
    CORPUS=${CORPUS_DIR}/wiki18_mini.jsonl
    SAVE_DIR=/home/boyuz5/data/indices/e5_Flat_mini
    echo "[$(date)] 构建 MINI 索引 (100K 语料) -> ${SAVE_DIR}"
else
    CORPUS=${CORPUS_DIR}/wiki18_100w_clean.jsonl
    SAVE_DIR=/home/boyuz5/data/indices/e5_Flat
    echo "[$(date)] 构建全量索引 (17.3M 语料) -> ${SAVE_DIR}"
fi

mkdir -p ${SAVE_DIR}

CUDA_VISIBLE_DEVICES=0 conda run -n rag python3 ${INDEX_BUILDER} \
    --retrieval_method e5 \
    --model_path ${MODEL} \
    --corpus_path ${CORPUS} \
    --save_dir ${SAVE_DIR} \
    --max_length 180 \
    --batch_size 512 \
    --use_fp16 \
    --faiss_type Flat \
    --pooling_method mean

echo "[$(date)] 索引构建完成: ${SAVE_DIR}"
ls -lh ${SAVE_DIR}/
