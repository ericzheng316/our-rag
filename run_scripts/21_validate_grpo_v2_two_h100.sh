#!/usr/bin/env bash
# Two-H100 validation harness for GRPO-v2.  It does not start or modify the
# retriever: run the retriever separately on RETRIEVER_GPU (normally GPU 1),
# then use this script to verify serial-vs-batch retrieval and sweep vLLM's
# memory fraction on TRAIN_GPU (normally GPU 0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/R3-RAG-Qwen}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"
TRAIN_GPU="${TRAIN_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/logs/grpo_v2_validation_$(date +%Y%m%d_%H%M%S)}"

VLLM_FRACTIONS="${VLLM_FRACTIONS:-0.45 0.50 0.55 0.60}"
NUM_EPISODES="${NUM_EPISODES:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
LORA_RANK="${LORA_RANK:-64}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
RETRIEVAL_BENCHMARK_QUERIES="${RETRIEVAL_BENCHMARK_QUERIES:-32}"
RUN_RETRIEVAL_BENCHMARK="${RUN_RETRIEVAL_BENCHMARK:-1}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
USE_ACEC="${USE_ACEC:-1}"
ACEC_ARTIFACT_V2="${ACEC_ARTIFACT_V2:-}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "Refusing to run: $REPO_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python runtime not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Training data not found: $TRAIN_DATA" >&2
    exit 2
fi
if [[ "$TRAIN_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "TRAIN_GPU and RETRIEVER_GPU must identify two different cards" >&2
    exit 2
fi
if [[ "$USE_ACEC" == "1" && -z "$ACEC_ARTIFACT_V2" ]]; then
    echo "USE_ACEC=1 requires ACEC_ARTIFACT_V2=/path/to/acec_calibration_v2.json" >&2
    exit 2
fi
if [[ "$USE_ACEC" == "1" && ! -f "$ACEC_ARTIFACT_V2" ]]; then
    echo "ACEC calibration artifact not found: $ACEC_ARTIFACT_V2" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/summary.tsv"
echo -e "vllm_fraction\tstatus\tlog" > "$SUMMARY"

echo "[preflight] repo=$REPO_ROOT"
echo "[preflight] commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[preflight] train_gpu=$TRAIN_GPU retriever_gpu=$RETRIEVER_GPU"
echo "[preflight] retriever=$RETRIEVE_URL"
echo "[preflight] vllm_fractions=$VLLM_FRACTIONS"
nvidia-smi -i "$TRAIN_GPU" --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi -i "$RETRIEVER_GPU" --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    echo "[unit] checking estimator gradient/KL/reward invariants"
    PYTHONPATH="$REPO_ROOT/rag/train" \
        "$PYTHON" "$REPO_ROOT/rag/train/tests/test_grpo_estimator_v2.py"
    echo "[unit] checking calibration-v2 labels/artifact/K modes"
    PYTHONPATH="$REPO_ROOT/rag/src" \
        "$PYTHON" "$REPO_ROOT/rag/src/belief/acec/tests/test_calibration_v2.py"
fi

if [[ "$RUN_RETRIEVAL_BENCHMARK" == "1" ]]; then
    echo "[retrieval] validating serial and batched result equivalence"
    "$PYTHON" "$SCRIPT_DIR/validate_retrieval_batch.py" \
        --url "$RETRIEVE_URL" \
        --data_path "$TRAIN_DATA" \
        --num_queries "$RETRIEVAL_BENCHMARK_QUERIES" \
        --batch_sizes 8 16 32 64 \
        --topk 5 \
        --out "$OUTPUT_ROOT/retrieval_batch_benchmark.json" \
        2>&1 | tee "$OUTPUT_ROOT/retrieval_batch_benchmark.log"
fi

read -r -a fractions <<< "$VLLM_FRACTIONS"
MAX_SAMPLES=$((NUM_EPISODES * BATCH_SIZE))
for fraction in "${fractions[@]}"; do
    RUN_DIR="$OUTPUT_ROOT/vllm_${fraction}"
    mkdir -p "$RUN_DIR"
    args=(
        --model_path "$MODEL_PATH"
        --data_path "$TRAIN_DATA"
        --save_path "$RUN_DIR/ckpt"
        --max_samples "$MAX_SAMPLES"
        --num_episodes "$NUM_EPISODES"
        --batch_size "$BATCH_SIZE"
        --n_samples "$N_SAMPLES"
        --lora_rank "$LORA_RANK"
        --rollout_temperature "$ROLLOUT_TEMPERATURE"
        --max_old_logprob_mae 0.25
        --vllm_gpu_mem_frac "$fraction"
        --save_steps 999999
        --lambda_ans 1.0
        --lambda_cov 0.3
        --lambda_fmt 0.1
        --retrieval_cost 0.05
        --clip_eps 0.2
    )
    if [[ "$USE_ACEC" == "1" ]]; then
        args+=(
            --use_acec
            --acec_artifact_v2 "$ACEC_ARTIFACT_V2"
            --acec_k_mode fixed
            --acec_fixed_k 2
        )
    fi

    echo "[memory] starting vllm_gpu_mem_frac=$fraction"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$RUN_DIR/gpu_before.csv"
    set +e
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
        "$PYTHON" "$REPO_ROOT/rag/train/grpo_rsf_vllm.py" "${args[@]}" \
        2>&1 | tee "$RUN_DIR/train.log"
    status=${PIPESTATUS[0]}
    set -e
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$RUN_DIR/gpu_after.csv"

    if [[ "$status" -ne 0 ]]; then
        echo -e "$fraction\tFAIL($status)\t$RUN_DIR/train.log" >> "$SUMMARY"
        echo "[memory] fraction=$fraction failed; stopping higher-fraction sweep" >&2
        break
    fi
    echo -e "$fraction\tPASS\t$RUN_DIR/train.log" >> "$SUMMARY"
done

echo "Validation summary: $SUMMARY"
cat "$SUMMARY"
