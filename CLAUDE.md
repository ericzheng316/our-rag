# Project Context — Uncertainty-Aware RAG

## Research Goal
Reproduce Fudan NLP R3-RAG (arXiv:2505.23794) Table 1, then add BeliefState (Beta-Bernoulli BAMDP) as an improvement. Paper claim: belief-guided early stopping reduces retrieval steps while maintaining accuracy.

## Repo Structure
```
rag/                        # R3-RAG codebase (Fudan NLP)
  benchmark/R3-RAG/src/
    inference_new.py        # Main inference (--use_belief, --distractor_file)
    cal_metric.py           # Evaluation → results.json + results.csv
  src/belief/
    belief_state.py         # Beta-Bernoulli BeliefState (use .ret_quality not .alpha)
    obs_extractor.py        # E5Embedder, extract_observation()
    belief_prompt.py        # (not used in inference anymore)
run_scripts/
  06_run_full_baseline.sh   # Full HotpotQA distractor baseline (7405条)
  07_run_full_belief.sh     # Full HotpotQA distractor belief run
  08_calc_metrics_full.sh   # Metric calculation for full runs
  chunked_encode.py         # Chunk-encode wiki corpus → fp16 memmap (GPU, ~30 min on A100)
  build_index_from_emb.py   # Build FAISS index from memmap (currently SQ8, change to Flat on A100)
  prep_full_distractor.py   # Merge flashrag dev + HF distractor → dev_distractor.jsonl
实験.md                     # Experiment design doc
算法設計.md                  # BAMDP/POMDP formalism doc
```

## Path Configuration (IMPORTANT when migrating)

All hardcoded paths use `/home/boyuz5` as base. On a new machine, replace in bulk:

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
| `run_scripts/0[0-8]_*.sh` | all paths |
| `rag/benchmark/R3-RAG/src/inference_new.py` | sys.path.insert, E5Embedder path |
| `rag/src/belief/obs_extractor.py` | E5_MODEL_PATH (smoke test only) |

Expected directory layout on new machine:
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
        dev_distractor.jsonl    # run prep_full_distractor.py to generate
      retrieval-corpus/
        wiki18_100w_clean.jsonl # 17.3M passages, 12GB
    datasets/
      hotpotqa/distractor_jsonl/
        dev.jsonl               # HF original format (needed for prep script)
    indices/
      e5_full_emb/
        embeddings.bin          # 25GB fp16 memmap (run chunked_encode.py)
        meta.json
      e5_Flat/
        e5_Flat.index           # build from embeddings.bin (needs ≥80GB RAM)
  logs/                         # created automatically by run scripts
```

## Models needed (download from HuggingFace)
- `Fudan-DISC/R3-RAG-Qwen` → inference model
- `intfloat/e5-base-v2` → retrieval embedder
- `Qwen/Qwen2.5-7B-Instruct` → judge model for cal_metric.py

## Data needed
- HotpotQA distractor dev: `datasets` library or HF (`hotpot_qa`, `distractor` config)
- Wiki corpus: flashrag format `wiki18_100w_clean.jsonl` (17.3M passages, 12GB)
- After encoding: `data/indices/e5_full_emb/embeddings.bin` (25GB fp16 memmap)

## Completed Experiments (2026-04-23, ICRN H200)
HotpotQA distractor, 7405 samples, num_search=5, docs=10:

| Config | EM_proc | F1_proc | Judge | avg_docs |
|--------|---------|---------|-------|----------|
| baseline | 58.0% | 74.9% | 81.6% | 11.21 |
| belief doc-filter | 58.0% | 74.9% | 81.6% | 11.21 |

**Finding:** Belief doc-filtering had zero effect in distractor mode. Real retrieval needed for belief to work.

## Current Status
- Distractor experiments done
- Wiki corpus encoded → `embeddings.bin` (25GB, on ICRN)
- Moving to dedicated A100 for full Flat FAISS index + real retrieval experiments
- **Belief early stopping NOT yet implemented** — this is the next key task

## Next Tasks (in order)
1. **On A100**: re-encode corpus (or transfer embeddings.bin from ICRN), build CPU Flat FAISS index (`build_index_from_emb.py`, change SQ8 → Flat, needs ≥80GB RAM)
2. **Implement belief early stopping** in `inference_new.py`: after belief update, if `ret_quality > threshold` → mark record for forced answer, skip further retrieval
3. **Run real retrieval**: baseline (no belief) + belief early stopping on HotpotQA full dev
4. **Paper table**: compare EM_proc, F1_proc, Judge, avg_docs across configs

## Key Implementation Notes
- `BeliefState` has NO `.alpha`/`.beta` — use `belief.ret_quality` for E[θ_ret]
- Belief prefix injection was **removed** (was hurting -1~2pt)
- `--use_belief False` (default) = clean baseline, no BeliefState overhead
- Distractor mode: pseudo E5 scores computed via cosine(query_emb, doc_emb), docs sorted best-first
- `results.csv` now written alongside `results.json` after every eval run
