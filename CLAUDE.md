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

## Completed Experiments — Full Results Table

HotpotQA dev, 7405条, num_search=5, docs_per_turn=10

| Config | EM_proc | F1_proc | Judge | avg_docs | Log dir |
|--------|---------|---------|-------|----------|---------|
| Distractor baseline | 58.0% | 74.9% | 81.6% | 11.21 | r3rag-qwen-distractor-baseline |
| Distractor belief (th=0.70) | 58.0% | 74.9% | 81.6% | 11.21 | r3rag-qwen-distractor-belief |
| **Real baseline** | **43.2%** | **58.1%** | **62.8%** | **15.87** | r3rag-qwen-real-full-baseline |
| Real belief v1 | 35.1% | 48.6% | 52.9% | 16.53 | r3rag-qwen-real-full-belief |
| Real belief v2 | 41.9% | 56.5% | 61.3% | 15.90 | (fixed v1 bugs partially) |
| Real belief no_rerank | 43.5% | 58.1% | 62.9% | 15.87 | r3rag-qwen-real-belief-norerank |
| Real belief v3 | 43.1% | 57.6% | 62.5% | 15.59 | r3rag-qwen-real-belief-v3 |
| Real belief v3b | 43.4% | 58.0% | 62.6% | 15.88 | r3rag-qwen-real-belief-v3b |

**Distractor finding:** Belief doc-filtering had zero effect. Root cause: gold/distractor docs share the same topic → E5 scores nearly identical → `ret_quality` never crosses threshold.

**Real retrieval finding (2026-04-27):** All belief variants within ±0.3pt of baseline. BeliefState provides no measurable improvement.

### Why BeliefState Has No Effect (diagnosed 2026-04-27)

Root cause: **R3-RAG already does implicit early stopping.** The model generates `[OUTPUT]` when satisfied:

```
belief.step distribution across 7405 samples:
  step 0:  629 (8.5%)   ← answers immediately, no retrieval
  step 1: 2481 (33.5%)
  step 2: 3661 (49.4%)  ← typical 2-hop query resolved in 2 turns
  step 3:  432 (5.8%)
  step 4+: 202 (2.7%)
  Average actual turns: ~1.62   → avg_docs = 1.62 × 9.8 ≈ 15.87
```

Condition A (early stop when `ret_quality > threshold`) **never fires**:
```
ret_quality distribution: mean=0.667, p90=0.747, p99=0.748, max=0.866
threshold=0.92 → 0/7405 triggers
threshold=0.85 → 1/7405 triggers
```

The Beta-Bernoulli model saturates: with prior (1,1) and ≤5 turns of moderate observations,
`E[θ_ret]` is bounded by ~0.85. Threshold=0.92 is unachievable in practice.

Dynamic budget (extra turns) activates for 355/7405 (4.8%) samples (hard queries), but their
final ret_quality (mean=0.612) is lower than average — extra turns don't help them.

### Cross-turn Reranking is Always Harmful — Never Enable

| Method | EM_proc | Judge | avg_docs |
|--------|---------|-------|----------|
| no_rerank | 43.5% | 62.9% | 15.87 |
| score-based rerank (v2) | 41.9% | 61.3% | 15.90 |
| RRF rerank (v3) | 43.1% | 62.5% | 15.59 |

---

## Current Status (2026-04-27)

**Current machine: H100 on Vast.ai**

- [x] All distractor experiments (7405条 baseline + belief)
- [x] Flat FAISS index built → `~/data/indices/e5_Flat/e5_Flat.index` (~53GB)
- [x] Real baseline done → 43.2% EM / 62.8% Judge
- [x] All belief variants done (v1→v3b) — no improvement over baseline
- [x] BeliefState failure diagnosed: model has implicit early stopping; Condition A threshold unreachable
- [ ] **DECISION NEEDED**: next research direction (see options below)

---

## BeliefState — Ablation 保留

BeliefState 全部代码保留，用作消融实验对比：

```
--use_belief --no_rerank   # BeliefState ablation
(无 flag)                  # 纯 baseline
--use_hyde                 # HyDE second hop
```

消融表设计：
| Config | EM | F1 | Judge | avg_docs | 备注 |
|--------|----|----|-------|----------|------|
| Baseline | 43.2% | 58.1% | 62.8% | 15.87 | script 09 |
| + BeliefState | 43.5% | 58.1% | 62.9% | 15.87 | script 10b (no_rerank) |
| + HyDE | 41.7% | 56.2% | 61.5% | 15.80 | script 10e ← -1.5pt，失败 |
| + GRPO R_sf | ? | ? | ? | ? | 待实现 |
| + HyDE + GRPO | ? | ? | ? | ? | 终极目标 |

---

## Research Roadmap (2026-04-27 确定方向)

**核心发现：** 桥接实体注入 sq2 与否，EM 差距 +27.9pt（71.1% vs 43.2%）。
瓶颈是检索召回率（49.5%）和 R3-RAG-Qwen 的 second-hop query 质量，不是 BeliefState。

### Phase 1 — HyDE（已实现，已运行，**失败 -1.5pt**）

**结果：** EM 41.7% vs baseline 43.2%，-1.5pt。

**失败根因：**
1. Hypothetical passage 凭 sub-query 字符串生成，没有第一跳检索证据做依据
   → 正确做法：基于 turn1 retrieved docs 生成 hypothetical，但这需要 R3-RAG-Qwen 生成
2. 错误级联放大：turns=2 → -3.4pt；turns=3 → -13.8pt；turns=4 → -15.7pt
3. 大多数 sq2 已实体化（"What govt position did Shirley Temple hold?"），HyDE 净加噪声

**结论：** HyDE 对 single-hop 有效，对 multi-hop 有害。此路不通，跳过。

**实现位置（代码保留备用）：**
- `split_server.py`: `/hyde_passage` endpoint
- `inference_new.py`: `--use_hyde` flag + `hyde_query_remote()`

### Phase 2 — GRPO with R_sf Process Reward

```
R_total = λ1·R_ans + λ2·Σ_t r_sf_marginal_t + λ3·R_format
r_sf_marginal_t = |new_sf_titles_found_at_t| / |sf_titles_total|
```

- 过程奖励直接对每次检索动作提供 dense 信号
- 教模型生成"桥接实体显式化"的 new_query
- 训练数据：HotpotQA train split（90k条，含 supporting_facts）

**数据准备：**
```bash
python3 -c "
from datasets import load_dataset; import json
ds = load_dataset('hotpot_qa', 'fullwiki', split='train')
with open('/root/data/flashrag_datasets/hotpotqa/train_sf.jsonl','w') as f:
    for r in ds:
        f.write(json.dumps({'id':r['id'],'question':r['question'],
          'golden_answers':[r['answer']],'supporting_facts':r['supporting_facts'],
          'type':r['type']}) + '\n')
"
```

### Phase 3 — Cold Start SFT 实体化（可选，配合 GRPO）

改 SFT 训练数据生成 prompt，让 teacher（GPT-4o/Claude）产出的轨迹中，
turn 2 的 new_query 必须包含 turn 1 检索到的桥接实体：
`"What government position did [Shirley Temple] hold?"` 而非 `"What was her role?"`

目标：给 GRPO 更好的 cold start，缩小策略探索空间。

---

## Key Diagnostic Numbers (2026-04-27)

```
bridge 问题占比:         80% (5918/7405)
桥接实体注入 sq2 率:     42% (1423/3358)
注入时 EM:              71.1%   未注入时 EM: 43.2%   差距: +27.9pt
检索命中率 (gold in docs): 49.5%  命中时 EM: 77.4%
当前整体 EM_proc:         43.2%  理论上界: ~77%
```

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
