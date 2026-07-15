#!/usr/bin/env bash
# Start the full retriever on a dedicated H100 with a source-built SM90 FAISS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/data}"
GPU_ID="${GPU_ID:-1}"
FAISS_GPU_USE_FLOAT16="${FAISS_GPU_USE_FLOAT16:-0}"
PORT="${PORT:-8001}"
HOST="${HOST:-$(hostname -I | awk '{print $1}')}"

FAISS_GPU_PYTHONPATH="${FAISS_GPU_PYTHONPATH:-${DATA_ROOT}/python_packages/faiss-gpu-sm90-1.14.3}"
MODEL="${MODEL:-${DATA_ROOT}/models/e5-base-v2}"
INDEX="${INDEX:-${DATA_ROOT}/indices/e5_Flat/e5_Flat.index}"
CORPUS="${CORPUS:-${DATA_ROOT}/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/rag/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
LOG="${LOG_DIR}/retriever_gpu_sm90.log"
ENV_FILE="${SCRIPT_DIR}/.env_retriever"

for path in "${FAISS_GPU_PYTHONPATH}" "${MODEL}" "${INDEX}" "${CORPUS}" "${PYTHON_BIN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"
{
    echo "HOST=${HOST}"
    echo "SPLIT_HOST=${HOST}"
    echo "RETRIEVE_URL=http://${HOST}:${PORT}/search"
} > "${ENV_FILE}"

echo "[$(date)] Starting paged SM90 GPU retriever: http://${HOST}:${PORT}"
echo "[$(date)] GPU_ID=${GPU_ID} FAISS_GPU_USE_FLOAT16=${FAISS_GPU_USE_FLOAT16} INDEX=${INDEX}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${FAISS_GPU_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
FAISS_GPU=1 \
FAISS_GPU_USE_FLOAT16="${FAISS_GPU_USE_FLOAT16}" \
"${PYTHON_BIN}" \
    "${PROJECT_ROOT}/rag/benchmark/retriever/src/retrive_server_gpu_paged.py" \
    --host "${HOST}" \
    --port "${PORT}" \
    --model_path "${MODEL}" \
    --index_path "${INDEX}" \
    --corpus_path "${CORPUS}" \
    2>&1 | tee -a "${LOG}"
