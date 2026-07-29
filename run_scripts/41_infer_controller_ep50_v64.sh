#!/usr/bin/env bash
# ACEC v6.4 policy x controller x verifier inference on two-card instances.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/root/data}"
PYTHON="${PYTHON:-$REPO_ROOT/rag/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PERSIST_ROOT/models/R3-RAG-Qwen}"
DATA_PATH="${DATA_PATH:-$PERSIST_ROOT/flashrag_datasets/hotpotqa/dev_distractor.jsonl}"
ACEC_ARTIFACT_V5="${ACEC_ARTIFACT_V5:-$PERSIST_ROOT/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$PERSIST_ROOT/models/e5-base-v2}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8001/search}"
INFERENCE_GPU="${INFERENCE_GPU:-0}"
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.55}"

ACEC_ADAPTER="${ACEC_ADAPTER:?set ACEC_ADAPTER to the ep50 adapter directory}"
ACEC_POLICY_NAME="${ACEC_POLICY_NAME:-acec_ep50}"
OUTCOME_ADAPTER="${OUTCOME_ADAPTER:-}"
OUTCOME_POLICY_NAME="${OUTCOME_POLICY_NAME:-outcome_ep75}"
VERIFIER_ARTIFACT="${VERIFIER_ARTIFACT:-}"
CONTROLLER_ARTIFACT="${CONTROLLER_ARTIFACT:-}"
SLOT_MANIFEST="${SLOT_MANIFEST:-}"
ORACLE_ENTITY_MANIFEST="${ORACLE_ENTITY_MANIFEST:-}"
QUESTION_IDS="${QUESTION_IDS:-}"
RUN_PROFILE="${RUN_PROFILE:-b0}"

case "$RUN_PROFILE" in
    b0)
        profile_controllers="none"
        profile_verifiers="first,majority,nli,acec_zero_shot"
        profile_samples="8"
        profile_k="1,2,4,8"
        profile_temperature="0.7"
        profile_preseed="0"
        ;;
    a0)
        profile_controllers="none,gold_entity"
        profile_verifiers="first"
        profile_samples="1"
        profile_k="1"
        profile_temperature="0.0"
        profile_preseed="0"
        ;;
    e5_off)
        profile_controllers="none"
        profile_verifiers="first"
        profile_samples="1"
        profile_k="1"
        profile_temperature="0.0"
        profile_preseed="0"
        ;;
    e5_on)
        profile_controllers="monitor_preseed"
        profile_verifiers="first"
        profile_samples="1"
        profile_k="1"
        profile_temperature="0.0"
        profile_preseed="1"
        ;;
    b1)
        profile_controllers="none"
        profile_verifiers="first,majority,nli,acec_zero_shot,acec_calibrated"
        profile_samples="8"
        profile_k="1,2,4,8"
        profile_temperature="0.7"
        profile_preseed="1"
        ;;
    a1)
        profile_controllers="none,belief_gap"
        profile_verifiers="first,nli,acec_zero_shot"
        profile_samples="1"
        profile_k="1"
        profile_temperature="0.0"
        profile_preseed="1"
        ;;
    c)
        profile_controllers="none,belief_adaptive"
        profile_verifiers="first"
        profile_samples="1"
        profile_k="1"
        profile_temperature="0.0"
        profile_preseed="1"
        ;;
    full)
        profile_controllers="none,monitor_preseed,belief_gap,belief_adaptive,belief_gap_adaptive"
        profile_verifiers="first,majority,nli,acec_zero_shot,acec_calibrated"
        profile_samples="8"
        profile_k="1,2,4,8"
        profile_temperature="0.7"
        profile_preseed="1"
        ;;
    *)
        echo "[v6.4 preflight] unsupported RUN_PROFILE=$RUN_PROFILE" >&2
        exit 2
        ;;
esac

CONTROLLERS="${CONTROLLERS:-$profile_controllers}"
VERIFIER_MODES="${VERIFIER_MODES:-$profile_verifiers}"

MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-$profile_samples}"
K_VALUES="${K_VALUES:-$profile_k}"
N_TURNS="${N_TURNS:-5}"
TEMPERATURE="${TEMPERATURE:-$profile_temperature}"
SEED="${SEED:-20260723}"
MAX_DOCS="${MAX_DOCS:-10}"
PRESEED_SLOTS="${PRESEED_SLOTS:-$profile_preseed}"
SLOT_K="${SLOT_K:-2}"
SLOT_PARSE_MIN_COVERAGE="${SLOT_PARSE_MIN_COVERAGE:-1.0}"
MIN_RETRIEVALS="${MIN_RETRIEVALS:-1}"
MAX_RETRIEVALS="${MAX_RETRIEVALS:-5}"
INJECTION_AFTER_RETRIEVALS="${INJECTION_AFTER_RETRIEVALS:-1}"
STOP_COVERAGE_MIN="${STOP_COVERAGE_MIN:-0.75}"
STOP_STD_MAX="${STOP_STD_MAX:-0.25}"
QUERY_STRATEGY="${QUERY_STRATEGY:-slot_plus_entity}"
BLOCK_LOW_COVERAGE_ANSWER="${BLOCK_LOW_COVERAGE_ANSWER:-1}"
NLI_BATCH_SIZE="${NLI_BATCH_SIZE:-128}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/logs/acec_v64_${RUN_PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ -e "$REPO_ROOT/.git/index.lock" ]]; then
    echo "[v6.4 preflight] refusing to run while .git/index.lock exists" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "[v6.4 preflight] Python runtime is not executable: $PYTHON" >&2
    exit 2
fi
for path in "$MODEL_PATH" "$DATA_PATH" "$ACEC_ARTIFACT_V5" "$E5_MODEL_PATH" "$ACEC_ADAPTER"; do
    if [[ ! -e "$path" ]]; then
        echo "[v6.4 preflight] required input is missing: $path" >&2
        exit 2
    fi
done
if [[ "$INFERENCE_GPU" == "$RETRIEVER_GPU" ]]; then
    echo "[v6.4 preflight] inference and retriever must use different GPUs" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[v6.4 preflight] refusing to overwrite: $OUTPUT_DIR" >&2
    exit 2
fi
if [[ -n "$VERIFIER_ARTIFACT" && ! -s "$VERIFIER_ARTIFACT" ]]; then
    echo "[v6.4 preflight] verifier artifact is missing/empty" >&2
    exit 2
fi
if [[ -n "$CONTROLLER_ARTIFACT" && ! -s "$CONTROLLER_ARTIFACT" ]]; then
    echo "[v6.4 preflight] controller artifact is missing/empty" >&2
    exit 2
fi
if [[ "$VERIFIER_MODES" == *"acec_calibrated"* && -z "$VERIFIER_ARTIFACT" ]]; then
    echo "[v6.4 preflight] acec_calibrated requires VERIFIER_ARTIFACT" >&2
    exit 2
fi
if [[ "$RUN_PROFILE" == "c" || "$RUN_PROFILE" == "full" ]] && [[ -z "$CONTROLLER_ARTIFACT" ]]; then
    echo "[v6.4 preflight] profile $RUN_PROFILE requires a cost-matched CONTROLLER_ARTIFACT" >&2
    exit 2
fi
if [[ "$RUN_PROFILE" == "full" && -z "$OUTCOME_ADAPTER" ]]; then
    echo "[v6.4 preflight] profile full requires the matched OUTCOME_ADAPTER" >&2
    exit 2
fi
if [[ "$CONTROLLERS" == *"gold_entity"* && -z "$ORACLE_ENTITY_MANIFEST" ]]; then
    echo "[v6.4 preflight] gold_entity controller requires ORACLE_ENTITY_MANIFEST" >&2
    exit 2
fi

echo "[v6.4 preflight] profile=$RUN_PROFILE commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[v6.4 preflight] questions=$MAX_QUESTIONS samples=$N_SAMPLES K=$K_VALUES"
echo "[v6.4 preflight] controllers=$CONTROLLERS verifier_modes=$VERIFIER_MODES"
echo "[v6.4 preflight] preseed=$PRESEED_SLOTS slot_k=$SLOT_K seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

PYTHONPATH="$REPO_ROOT/rag/src" "$PYTHON" -c \
    'import sys; from belief.acec.calibration_v5 import load_calibration_artifact_v5; a=load_calibration_artifact_v5(sys.argv[1]); assert a.metrics["gate"]["pass"] is True; print("[v6.4 preflight] ACEC v5 artifact gate=PASS")' \
    "$ACEC_ARTIFACT_V5"

RETRIEVE_URL="$RETRIEVE_URL" "$PYTHON" -c \
    'import os, requests; r=requests.post(os.environ["RETRIEVE_URL"], json={"querys":["ACEC v6.4 health check"]}, timeout=120); r.raise_for_status(); data=r.json(); assert isinstance(data, list) and len(data)==1; print("[v6.4 preflight] retriever=PASS docs="+str(len(data[0])))'

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    PYTHONPATH="$REPO_ROOT/rag/src:$REPO_ROOT/rag/train" \
        "$PYTHON" -m unittest \
        rag.train.tests.test_inference_controller_v64 \
        rag.train.tests.test_answer_verifier_v64 -v
fi

args=(
    --model_path "$MODEL_PATH"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --run_label "$RUN_PROFILE"
    --acec_artifact_v5 "$ACEC_ARTIFACT_V5"
    --adapter "$ACEC_POLICY_NAME=$ACEC_ADAPTER"
    --e5_model_path "$E5_MODEL_PATH"
    --acec_nli_model "$NLI_MODEL"
    --retrieve_url "$RETRIEVE_URL"
    --max_questions "$MAX_QUESTIONS"
    --question_batch_size "$QUESTION_BATCH_SIZE"
    --n_samples "$N_SAMPLES"
    --k_values "$K_VALUES"
    --verifier_modes "$VERIFIER_MODES"
    --n_turns "$N_TURNS"
    --temperature "$TEMPERATURE"
    --seed "$SEED"
    --max_docs "$MAX_DOCS"
    --preseed_slots "$PRESEED_SLOTS"
    --slot_k "$SLOT_K"
    --slot_parse_min_coverage "$SLOT_PARSE_MIN_COVERAGE"
    --min_retrievals "$MIN_RETRIEVALS"
    --max_retrievals "$MAX_RETRIEVALS"
    --injection_after_retrievals "$INJECTION_AFTER_RETRIEVALS"
    --stop_coverage_min "$STOP_COVERAGE_MIN"
    --stop_std_max "$STOP_STD_MAX"
    --query_strategy "$QUERY_STRATEGY"
    --block_low_coverage_answer "$BLOCK_LOW_COVERAGE_ANSWER"
    --nli_batch_size "$NLI_BATCH_SIZE"
    --bootstrap_iterations "$BOOTSTRAP_ITERATIONS"
    --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC"
)
if [[ -n "$OUTCOME_ADAPTER" ]]; then
    args+=(--adapter "$OUTCOME_POLICY_NAME=$OUTCOME_ADAPTER")
fi
IFS=',' read -r -a controller_values <<< "$CONTROLLERS"
for controller in "${controller_values[@]}"; do
    args+=(--controller "$controller")
done
if [[ -n "$VERIFIER_ARTIFACT" ]]; then
    args+=(--verifier_artifact "$VERIFIER_ARTIFACT")
fi
if [[ -n "$CONTROLLER_ARTIFACT" ]]; then
    args+=(--controller_artifact "$CONTROLLER_ARTIFACT")
fi
if [[ -n "$SLOT_MANIFEST" ]]; then
    args+=(--slot_manifest "$SLOT_MANIFEST")
fi
if [[ -n "$ORACLE_ENTITY_MANIFEST" ]]; then
    args+=(--oracle_entity_manifest "$ORACLE_ENTITY_MANIFEST")
fi
if [[ -n "$QUESTION_IDS" ]]; then
    args+=(--question_ids "$QUESTION_IDS")
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$INFERENCE_GPU" RETRIEVE_URL="$RETRIEVE_URL" \
    "$PYTHON" "$REPO_ROOT/rag/train/infer_belief_controller_v64.py" \
    "${args[@]}" 2>&1 | tee "${OUTPUT_DIR}.log"

test -s "$OUTPUT_DIR/results.json"
echo "[v6.4] COMPLETE: $OUTPUT_DIR/results.json"
