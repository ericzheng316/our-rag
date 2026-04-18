#!/bin/bash
# 准备 mini smoke test 数据：
#   - hotpotqa dev_mini.jsonl (前200条)
#   - wiki18_mini.jsonl (前100K段落)
#   用于快速跑通流程，不追求检索质量

set -e

DATASET_ROOT=/home/boyuz5/data/flashrag_datasets
CORPUS_DIR=${DATASET_ROOT}/retrieval-corpus
MINI_N_DEV=200
MINI_N_CORPUS=100000

echo "[$(date)] 准备 mini 数据集..."

# --- HotpotQA dev mini ---
SRC=${DATASET_ROOT}/hotpotqa/dev.jsonl
DST=${DATASET_ROOT}/hotpotqa/dev_mini.jsonl
if [ -f "${DST}" ]; then
    echo "  dev_mini.jsonl 已存在，跳过 ($(wc -l < ${DST}) 条)"
else
    head -n ${MINI_N_DEV} ${SRC} > ${DST}
    echo "  创建 ${DST} (${MINI_N_DEV} 条)"
fi

# --- Mini wiki corpus ---
SRC=${CORPUS_DIR}/wiki18_100w_clean.jsonl
DST=${CORPUS_DIR}/wiki18_mini.jsonl
if [ -f "${DST}" ]; then
    echo "  wiki18_mini.jsonl 已存在，跳过 ($(wc -l < ${DST}) 条)"
else
    head -n ${MINI_N_CORPUS} ${SRC} > ${DST}
    echo "  创建 ${DST} (${MINI_N_CORPUS} 条)"
fi

echo "[$(date)] 数据准备完成"
echo "  dev_mini: ${DATASET_ROOT}/hotpotqa/dev_mini.jsonl"
echo "  mini corpus: ${CORPUS_DIR}/wiki18_mini.jsonl"
