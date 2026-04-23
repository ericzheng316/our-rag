# Project Context — Uncertainty-Aware RAG

## Research Goal
Reproduce Fudan NLP R3-RAG (arXiv:2505.23794) Table 1, then add BeliefState (Beta-Bernoulli BAMDP) as an improvement. Paper claim: belief-guided early stopping reduces retrieval steps while maintaining accuracy.

---

## Quick Start on a New Machine

### Step 1 — Clone repo and fix paths

```bash
git clone https://github.com/<YOUR_GITHUB>/rag.git ~/rag
cd ~/rag

# Replace all /home/boyuz5 with your actual home dir (e.g. /home/ubuntu)
grep -rl "/home/boyuz5" run_scripts/ rag/benchmark/R3-RAG/src/inference_new.py rag/src/belief/obs_extractor.py \
  | xargs sed -i 's|/home/boyuz5|/home/ubuntu|g'
```

### Step 2 — Create the TWO Python environments

**This project uses two separate environments. Do not mix them.**

| Environment | Used for | Key packages |
|-------------|----------|-------------|
| `rag/.venv` (Python 3.13) | Inference, judge (vllm), split server | vllm, transformers, torch |
| `rag` conda env (Python 3.12) | Retriever server (faiss+flashrag) | faiss-gpu, flashrag, fastapi |

#### Environment A: `.venv` (vllm inference)

```bash
# From repo root (~/rag)
python3.13 -m venv .venv
.venv/bin/pip install vllm==0.18.1
# vllm will pull torch, transformers, etc. automatically
```

> vllm 0.18.1 requires CUDA 12.6+. Check driver: `nvidia-smi`.

#### Environment B: `rag` conda env (faiss retriever)

```bash
# faiss-gpu MUST come from conda, not pip
conda create -n rag python=3.12 -y
conda install -n rag -c conda-forge faiss-gpu=1.14.1 cuda-version=12.6 -y
conda run -n rag pip install fastapi uvicorn torch transformers

# flashrag: editable install from local copy in this repo (NOT from PyPI)
conda run -n rag pip install -e ~/rag/tool/FlashRAG
```

### Step 3 — Download models (HuggingFace)

```bash
# Run these from ~/  (models land in ~/models/)
mkdir -p ~/models

# Inference model (R3-RAG fine-tuned Qwen)
huggingface-cli download Fudan-DISC/R3-RAG-Qwen --local-dir ~/models/R3-RAG-Qwen

# Retrieval embedder
huggingface-cli download intfloat/e5-base-v2 --local-dir ~/models/e5-base-v2

# Judge model for cal_metric.py (Qwen2.5-7B, NOT 72B)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
```

### Step 4 — Download data

```bash
mkdir -p ~/data/flashrag_datasets/hotpotqa
mkdir -p ~/data/flashrag_datasets/retrieval-corpus
mkdir -p ~/data/datasets/hotpotqa/distractor_jsonl

# (A) Wiki corpus — from FlashRAG HuggingFace data release
# https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
# File: retrieval-corpus/wiki18_100w_clean.jsonl (12GB, 17.3M passages)
huggingface-cli download RUC-NLPIR/FlashRAG_datasets \
  --repo-type dataset \
  --include "retrieval-corpus/wiki18_100w_clean.jsonl" \
  --local-dir ~/data/flashrag_datasets

# (B) HotpotQA flashrag format dev split
# File: hotpotqa/dev.jsonl (7405 samples, flashrag format)
huggingface-cli download RUC-NLPIR/FlashRAG_datasets \
  --repo-type dataset \
  --include "hotpotqa/dev.jsonl" \
  --local-dir ~/data/flashrag_datasets

# (C) HotpotQA original HF format (needed to build dev_distractor.jsonl)
python3 -c "
from datasets import load_dataset
ds = load_dataset('hotpot_qa', 'distractor', split='validation')
import os, json
os.makedirs('data/datasets/hotpotqa/distractor_jsonl', exist_ok=True)
with open('data/datasets/hotpotqa/distractor_jsonl/dev.jsonl', 'w') as f:
    for row in ds:
        f.write(json.dumps(row) + '\n')
print('Done:', len(ds), 'samples')
"

# (D) Build dev_distractor.jsonl (merges flashrag + HF formats)
conda run -n rag python3 ~/rag/run_scripts/prep_full_distractor.py
# Output: ~/data/flashrag_datasets/hotpotqa/dev_distractor.jsonl
```

### Step 5 — Encode corpus (GPU, ~30 min on A100)

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rag python3 ~/rag/run_scripts/chunked_encode.py
# Output: ~/data/indices/e5_full_emb/embeddings.bin  (25GB, fp16)
#          ~/data/indices/e5_full_emb/meta.json
```

### Step 6 — Build FAISS Flat index (CPU, needs ≥80GB RAM)

```bash
# Uses ~/data/indices/e5_full_emb/embeddings.bin
# Outputs ~/data/indices/e5_Flat/e5_Flat.index  (~53GB)
conda run -n rag python3 ~/rag/run_scripts/build_index_from_emb.py
```

> **RAM requirement:** Flat index for 17.3M×768 fp32 ≈ 53GB. Need ≥80GB total RAM.
> If the machine has <80GB RAM, use a server with more memory or reduce to SQ8 (but accuracy drops ~1pt).

---

## Running Experiments

### Distractor mode (no retrieval server needed)

Already completed on ICRN. Scripts 06/07/08 cover this.

```bash
bash run_scripts/06_run_full_baseline.sh        # baseline (7405条)
bash run_scripts/07_run_full_belief.sh           # belief doc-filter (7405条)
bash run_scripts/08_calc_metrics_full.sh         # baseline metrics
BELIEF=1 bash run_scripts/08_calc_metrics_full.sh  # belief metrics
```

### Real retrieval mode (FAISS index required)

Start services first (each in its own terminal or tmux pane):

```bash
# Terminal 1 — Retriever server (conda rag env, uses faiss-gpu)
bash run_scripts/02_start_retriever.sh

# Terminal 2 — Split query server (vllm, Qwen2.5-7B)
bash run_scripts/03_start_split_server.sh

# Terminal 3 — Inference (wait until both servers are ready)
bash run_scripts/09_run_real_baseline.sh         # baseline
# OR
bash run_scripts/10_run_real_belief.sh           # belief early-stopping

# After inference finishes, calculate metrics:
bash run_scripts/11_calc_metrics_real.sh
BELIEF=1 bash run_scripts/11_calc_metrics_real.sh
```

---

## Repo Structure

```
rag/                        # R3-RAG codebase (Fudan NLP)
  .venv/                    # Python 3.13 venv (vllm inference) — NOT committed
  benchmark/R3-RAG/src/
    inference_new.py        # Main inference (--use_belief, --distractor_file)
    cal_metric.py           # Evaluation → results.json + results.csv
    split_server.py         # Query rewrite server (vllm, Qwen2.5-7B)
  benchmark/retriever/src/
    retrive_server.py       # Dense retrieval server (flashrag + faiss)
  src/belief/
    belief_state.py         # Beta-Bernoulli BeliefState (use .ret_quality not .alpha)
    obs_extractor.py        # E5Embedder, extract_observation()
  tool/FlashRAG/            # flashrag editable install source
run_scripts/
  02_start_retriever.sh     # Start E5 retriever server (conda rag env)
  03_start_split_server.sh  # Start split query server (vllm .venv)
  06_run_full_baseline.sh   # Distractor baseline (7405条, no server needed)
  07_run_full_belief.sh     # Distractor belief doc-filter (7405条)
  08_calc_metrics_full.sh   # Metrics for distractor runs
  09_run_real_baseline.sh   # Real retrieval baseline (7405条)
  10_run_real_belief.sh     # Real retrieval belief early-stopping (7405条)
  11_calc_metrics_real.sh   # Metrics for real retrieval runs
  chunked_encode.py         # Chunk-encode wiki corpus → fp16 memmap (GPU)
  build_index_from_emb.py   # Build Flat FAISS index from memmap (CPU, ≥80GB RAM)
  prep_full_distractor.py   # Merge flashrag dev + HF distractor → dev_distractor.jsonl
实験.md                     # Experiment design doc
算法設計.md                  # BAMDP/POMDP formalism doc
```

---

## Path Configuration (migrating to new machine)

All hardcoded paths use `/home/boyuz5` as base. Replace in bulk:

```bash
# Run from repo root, replace /home/boyuz5 with your actual home dir
grep -rl "/home/boyuz5" run_scripts/ rag/benchmark/R3-RAG/src/inference_new.py rag/src/belief/obs_extractor.py \
  | xargs sed -i 's|/home/boyuz5|/YOUR/HOME|g'
```

Files with hardcoded paths:

| File | What changes |
|------|-------------|
| `run_scripts/chunked_encode.py` | CORPUS_PATH, MODEL_PATH, SAVE_DIR |
| `run_scripts/build_index_from_emb.py` | EMB_DIR, INDEX_PATH |
| `run_scripts/prep_full_distractor.py` | FLASHRAG_PATH, HF_PATH, OUTPUT_PATH |
| `run_scripts/0[0-9]_*.sh` | all paths |
| `rag/benchmark/R3-RAG/src/inference_new.py` | sys.path.insert, E5Embedder path |
| `rag/src/belief/obs_extractor.py` | E5_MODEL_PATH (smoke test only) |

Expected directory layout:

```
$HOME/
  rag/                          # git clone of this repo
  models/
    R3-RAG-Qwen/                # HF: Fudan-DISC/R3-RAG-Qwen
    e5-base-v2/                 # HF: intfloat/e5-base-v2
    Qwen2.5-7B-Instruct/        # HF: Qwen/Qwen2.5-7B-Instruct
  data/
    flashrag_datasets/
      hotpotqa/
        dev.jsonl               # flashrag format (7405条)
        dev_distractor.jsonl    # generated by prep_full_distractor.py
      retrieval-corpus/
        wiki18_100w_clean.jsonl # 17.3M passages, 12GB (from FlashRAG HF dataset)
    datasets/
      hotpotqa/distractor_jsonl/
        dev.jsonl               # HF original format (needed for prep script)
    indices/
      e5_full_emb/
        embeddings.bin          # 25GB fp16 memmap (from chunked_encode.py)
        meta.json
      e5_Flat/
        e5_Flat.index           # ~53GB Flat index (from build_index_from_emb.py)
  logs/                         # created automatically
```

---

## Completed Experiments (2026-04-23, ICRN H200, distractor mode)

HotpotQA distractor, 7405 samples, num_search=5, docs=10:

| Config | EM_proc | F1_proc | Judge | avg_docs |
|--------|---------|---------|-------|----------|
| baseline (no belief) | 58.0% | 74.9% | 81.6% | 11.21 |
| belief doc-filter (th=0.70) | 58.0% | 74.9% | 81.6% | 11.21 |

**Finding:** Belief doc-filtering had zero effect in distractor mode.
Root cause: In distractor setting, gold and distractor docs share the same topic → E5 cosine scores are nearly identical → `ret_quality` never crosses 0.70 → threshold never triggers.
**Real retrieval is required for belief to show improvement.**

---

## Current Status (2026-04-23)

- [x] Distractor experiments done (7405条 baseline + belief)
- [x] Wiki corpus encoded → `embeddings.bin` 25GB (on ICRN H200)
- [x] CLAUDE.md written for context persistence
- [ ] **Migrate to dedicated A100** (ICRN shared, only ~32GB CPU RAM free)
- [ ] **Build Flat FAISS index** (needs ≥80GB RAM, run `build_index_from_emb.py`)
- [ ] **Implement belief early stopping** in `inference_new.py` (NOT YET DONE)
- [ ] **Run real retrieval experiments** (09/10/11 scripts ready, index not built yet)

---

## Next Tasks (in order)

1. **Rent dedicated A100** (Lambda Labs / RunPod / AutoDL):
   - A100 80GB SXM typically has 512GB+ system RAM → enough for Flat index build
   - Re-encode corpus (30 min) OR transfer `embeddings.bin` via `rsync` from ICRN
   - Build Flat FAISS index: `conda run -n rag python3 run_scripts/build_index_from_emb.py`

2. **Implement belief early stopping** in `inference_new.py`:
   - After each retrieval round, update BeliefState
   - If `belief.ret_quality > threshold` → set a flag → force-answer pass (no more retrieval)
   - Key paper claim: reduces `avg_docs` while maintaining `EM_proc`/`Judge`

3. **Run real retrieval** (09_run_real_baseline.sh + 10_run_real_belief.sh)

4. **Paper table**: compare distractor-baseline vs real-baseline vs real-belief-early-stop

---

## Key Implementation Notes

- `BeliefState` has NO `.alpha`/`.beta` attributes — use `belief.ret_quality` for E[θ_ret]
- Belief prefix injection was **removed** (was hurting -1~2pt on distractor)
- `--use_belief False` (default) = clean baseline; E5Embedder is **not loaded** in baseline (no VRAM waste)
- `--e5_model_path` in run scripts: path to e5-base-v2 model, defaults to `$HOME/models/e5-base-v2` if omitted
- `STOP_TOKEN_ID=151645` is Qwen2.5 EOS token — do not change for R3-RAG-Qwen
- `--tp 1` (default): tensor_parallel for vllm; increase to match GPU count (e.g., `--tp 4` on 4×A100)
- Retriever server writes `HOST`/`SPLIT_HOST` to `run_scripts/.env_retriever`; inference scripts `source` this file — **start retriever before inference**
- `results.csv` is written alongside `results.json` after every eval run
- `faiss_gpu=False` in `retrive_server.py` — CPU FAISS search. Flat index (53GB) cannot fit on any single GPU alongside the LLM; CPU retrieval is required
- The `.venv` is **not committed** to git. Recreate with `python3.13 -m venv .venv && .venv/bin/pip install vllm==0.18.1`
- Interrupted inference can be resumed: `solve_init()` checks for existing `records.jsonl` and loads completed records, skipping already-finished samples
- `--split_url` is required for real retrieval mode (the split server rewrites compound questions into sub-queries). Without it, R3-RAG's multi-hop decomposition breaks silently

## GPU / RAM Requirements

| Process | VRAM | CPU RAM | Script |
|---------|------|---------|--------|
| vllm inference (R3-RAG-Qwen ~7B, tp=1) | ~16GB | - | 06/07/09/10 |
| E5 retrieval embedder (when use_belief) | ~0.5GB | - | 07/10 |
| Retriever FAISS search (CPU) | 0 | ~55GB (index loaded to RAM) | 02 |
| Split server (Qwen2.5-7B, tp=1) | ~16GB | - | 03 |
| Build Flat FAISS index | 0 | ≥80GB | build_index_from_emb.py |

**Practical setup on single A100 (80GB VRAM, 512GB RAM):**
- GPU 0: vllm inference (R3-RAG-Qwen)
- GPU 1 (or same GPU): split server (Qwen2.5-7B) — or `--tp 1` on same GPU if 80GB fits both
- Retriever: CPU-only (FAISS Flat in system RAM ~53GB)
