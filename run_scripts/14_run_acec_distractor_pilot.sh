#!/bin/bash
# ACEC Week-1 pilot: small-subset distractor-mode inference producing a fresh
# records.jsonl that carries 'supporting_facts' (requires the
# prep_full_distractor.py + inference_new.py fixes that add this field —
# regenerate dev_distractor.jsonl with prep_full_distractor.py first if your
# copy predates it).
#
# Distractor mode needs ONLY the .venv (vllm) — no retriever server, no FAISS
# index, no conda `rag` env. This matters on RAM-constrained boxes (e.g. a
# 32GB-CPU-RAM cluster node): the real-retrieval Flat index needs ~53GB just
# to load, so real retrieval mode is infeasible there regardless of GPU size,
# while this pilot only needs ~16GB VRAM for R3-RAG-Qwen.
#
# NUM_DOCS matters a lot here: a HotpotQA distractor question has exactly 10
# candidate paragraphs (2 gold + 8 distractor) total. If NUM_DOCS==10, every
# turn retrieves the entire pool regardless of query quality, so gold-hit
# labels become trivially "always hit" and the ACEC observation model has
# nothing to discriminate (empirically: pi_a saturates near 1.0 for every
# action mode, AUC becomes meaningless). Default here is 3, forcing genuine
# per-turn selection out of the 10 so hit/miss actually depends on the query.
#
# 用法: MAX_SAMPLES=500 NUM_DOCS=3 bash 14_run_acec_distractor_pilot.sh

set -e

MODEL=${MODEL:-$HOME/models/R3-RAG-Qwen}
STOP_TOKEN_ID=151645
DATASET_ROOT=${DATASET_ROOT:-$HOME/data/flashrag_datasets}
MODEL_NAME="r3rag-qwen-distractor-acec-pilot"
LOG_DIR=$HOME/logs/${MODEL_NAME}
DISTRACTOR_FILE="${DATASET_ROOT}/hotpotqa/dev_distractor.jsonl"
MAX_SAMPLES=${MAX_SAMPLES:-500}
NUM_DOCS=${NUM_DOCS:-3}

mkdir -p ${LOG_DIR}
echo "[$(date)] ACEC pilot: hotpotqa distractor, ${MAX_SAMPLES} samples, docs=${NUM_DOCS}, no belief"
echo "  log: ${LOG_DIR}/inference.log"

cd $HOME/rag/benchmark/R3-RAG

CUDA_VISIBLE_DEVICES=0 DATASET_ROOT=${DATASET_ROOT} \
$HOME/rag/.venv/bin/python src/inference_new.py \
    --model_path ${MODEL} \
    --log_dir ${LOG_DIR} \
    --num_search_one_attempt 5 \
    --stop_token_id ${STOP_TOKEN_ID} \
    --num_of_docs ${NUM_DOCS} \
    --tp 1 \
    --datasets hotpotqa \
    --dev_file dev_distractor.jsonl \
    --distractor_file ${DISTRACTOR_FILE} \
    --max_samples ${MAX_SAMPLES} \
    2>&1 | tee ${LOG_DIR}/inference.log

echo "[$(date)] 推理完成 -> ${LOG_DIR}/records.jsonl"
