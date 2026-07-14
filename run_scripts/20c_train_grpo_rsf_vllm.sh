#!/usr/bin/env bash
# GRPO + R_sf trainer, vLLM-accelerated rollout generation + lightweight
# HF+PEFT gradient step (rag/train/grpo_rsf_vllm.py). Single GPU.
#
# Why this exists over 20b_train_grpo_rsf_simple.sh: that script generates
# one rollout at a time via HF transformers.generate() — correct, but
# ~17s/rollout serial (measured: 32 rollouts, ~9 min). At design doc D11-14's
# smoke-run scale (100 steps, 5k prompts, G=8 ≈ 6400 rollouts) that's ~30
# hours on one GPU. This script batches rollout generation through vLLM
# (already confirmed both fast and correct for this model) instead, while
# keeping the exact same gradient-step logic as grpo_rsf_simple.py.
#
# Validated end to end 2026-07-13 (gold-SF reward mode) — see
# experiments/2026-07-13_grpo_rsf_vllm_first_validation/. --use_acec below
# (belief-shaped R_cov reward instead of gold-SF) is new and NOT yet run —
# see grpo_rsf_vllm.py's module docstring for what it needs and the
# unresolved question of where the Week-1 calibration artifact actually
# lives. Strongly do a tiny run first with any config change — see Usage —
# before trusting the full design-doc-scale defaults.
#
# Prerequisites:
#   1. Retriever server running:  bash run_scripts/02_start_retriever.sh
#   2. Training data downloaded (see 20_train_grpo_rsf.sh's header for the
#      download snippet — same train_sf.jsonl file, shared across all three
#      GRPO-RSF scripts).
#   3. Same rag/.venv as grpo_rsf_simple.py — no new dependencies for the
#      default gold-SF mode (vLLM was already installed for the Week-1/2
#      inference pipeline). --use_acec additionally needs
#      sentence-transformers (pip install -e rag/src/belief/acec[nli]).
#
# Usage:
#   # tiny correctness-only pass FIRST (do this before the full spec):
#   MAX_SAMPLES=20 NUM_EPISODES=3 N_SAMPLES=4 LORA_RANK=16 \
#     bash run_scripts/20c_train_grpo_rsf_vllm.sh
#   # design-doc smoke-run scale, once the above looks sane:
#   bash run_scripts/20c_train_grpo_rsf_vllm.sh
#   # ACEC belief-shaped reward instead of gold-SF (tiny pass first, same as above):
#   USE_ACEC=1 MAX_SAMPLES=20 NUM_EPISODES=3 N_SAMPLES=4 LORA_RANK=16 \
#     bash run_scripts/20c_train_grpo_rsf_vllm.sh

set -euo pipefail

source "$(dirname "$0")/.env_retriever" 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/rag"
MODEL_PATH="$HOME/models/R3-RAG-Qwen"
TRAIN_DATA="$HOME/data/flashrag_datasets/hotpotqa/train_sf.jsonl"
OUTPUT_DIR="$HOME/logs/grpo_rsf_vllm_$(date +%Y%m%d_%H%M%S)"

# ── Servers (set by .env_retriever or override here) ──────────────────────────
export RETRIEVE_URL="${RETRIEVE_URL:-http://127.0.0.1:8081/search}"

# ── Smoke-run knobs (design doc D11-14 defaults; override for a cheap dry run) ─
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
NUM_EPISODES="${NUM_EPISODES:-100}"
N_SAMPLES="${N_SAMPLES:-8}"        # G, rollouts per prompt
BATCH_SIZE="${BATCH_SIZE:-8}"      # questions per episode
LORA_RANK="${LORA_RANK:-64}"
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.45}"

# ── ACEC belief reward mode (off by default — gold-SF is the validated path) ───
USE_ACEC="${USE_ACEC:-0}"
E5_MODEL_PATH="${E5_MODEL_PATH:-$HOME/models/e5-base-v2}"
ACEC_NLI_MODEL="${ACEC_NLI_MODEL:-cross-encoder/nli-deberta-v3-base}"
# Empty by default on purpose: verify this file actually exists on this box
# before pointing at it (see grpo_rsf_vllm.py's module docstring) — omitting
# it still runs, just with uncalibrated ACECConfig() defaults + a warning.
ACEC_OBSERVATION_MODEL="${ACEC_OBSERVATION_MODEL:-}"

mkdir -p "$OUTPUT_DIR"

echo "[GRPO-RSF-vLLM] Output dir:  $OUTPUT_DIR"
echo "[GRPO-RSF-vLLM] Retriever:   $RETRIEVE_URL"
echo "[GRPO-RSF-vLLM] max_samples=$MAX_SAMPLES num_episodes=$NUM_EPISODES n_samples=$N_SAMPLES batch_size=$BATCH_SIZE lora_rank=$LORA_RANK vllm_gpu_mem_frac=$VLLM_GPU_MEM_FRAC use_acec=$USE_ACEC"

args=(
    --model_path "$MODEL_PATH"
    --data_path "$TRAIN_DATA"
    --save_path "$OUTPUT_DIR/ckpt"
    --max_samples "$MAX_SAMPLES"
    --num_episodes "$NUM_EPISODES"
    --n_samples "$N_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --lora_rank "$LORA_RANK"
    --vllm_gpu_mem_frac "$VLLM_GPU_MEM_FRAC"
)
if [[ "$USE_ACEC" == "1" ]]; then
    args+=(--use_acec --e5_model_path "$E5_MODEL_PATH" --acec_nli_model "$ACEC_NLI_MODEL")
    if [[ -n "$ACEC_OBSERVATION_MODEL" ]]; then
        args+=(--acec_observation_model "$ACEC_OBSERVATION_MODEL")
    fi
    # CrossEncoder(model_name) calls hf_hub_download to check/fetch the model
    # even when it's already cached (a network round-trip every single time,
    # regardless of cache hit) — HF_HOME must point at the actual persistent
    # cache (else it checks the wrong, empty, ephemeral ~/.cache/huggingface),
    # and confirmed cross-encoder/nli-deberta-v3-base is fully cached there
    # already, so HF_HUB_OFFLINE=1 skips the network round-trip entirely
    # instead of just rerouting it through the mirror.
    export HF_HOME="${HF_HOME:-$HOME/data/hf_cache}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"  # fallback if offline mode ever hits a cache miss
fi

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/train/grpo_rsf_vllm.py" \
    "${args[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
