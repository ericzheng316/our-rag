# Project Context — ACEC-Belief RAG

## Research Goal
Action-Conditioned Evidence-Coverage Belief (ACEC) for agentic multi-hop RAG,
built on an R3-RAG (arXiv:2505.23794) backbone. Full design: see
[ACEC_Belief_RAG_design.md](ACEC_Belief_RAG_design.md) (v1.0, 2026-07-02).

This supersedes the earlier system-level Beta-Bernoulli BeliefState approach
and its roadmap (HyDE, gold-SF-only GRPO, etc.) — those produced no measurable
improvement over the R3-RAG baseline and are not the current direction. The
old code is kept at `rag/src/belief/` (not `acec/`) purely as an ablation
baseline; do not treat its prior findings as guidance for new work.

## Current implementation
- `rag/src/belief/acec/` — ACEC belief module (action labeler, coverage
  filter, observation model, reward shaping, offline calibration). See its
  `__init__.py` and per-file docstrings for the API; `pyproject.toml` there
  lists dependencies (core: numpy; optional: torch/transformers, sentence-transformers).
- `run_scripts/build_acec_calibration.py` (+ `14_run_acec_distractor_pilot.sh`,
  `15_build_acec_calibration.sh`) — Week-1 calibration/go-no-go gate pipeline
  (design doc Section 9): replay a distractor-mode `records.jsonl` (now
  carrying `supporting_facts`, via the `prep_full_distractor.py` /
  `inference_new.py` fixes) through `ACECBeliefState` and check per-slot hit
  AUC / K-accuracy gates before investing further.
- `14_run_acec_distractor_pilot.sh`'s `NUM_DOCS` (default 3, not the usual 10):
  a HotpotQA distractor question has exactly 10 candidate paragraphs (2 gold +
  8 distractor). `--num_of_docs 10` retrieves the whole pool every turn
  regardless of query quality, making gold-hit labels trivially "always hit"
  (empirically: pi_a saturates ~1.0 per action mode, AUC becomes meaningless —
  first pilot run hit exactly this). Keep it below 10 so hit/miss actually
  depends on the query.
- **Week-1 gate: PASS** (2026-07-07, 500-sample HotpotQA distractor pilot,
  `cross-encoder/nli-deberta-v3-base`): held-out per-slot hit AUC 0.86-0.93
  across a `--tau_new` sweep, K-accuracy 1.00 (likely trivial on HotpotQA —
  nearly every question has exactly 2 supporting facts, so "always guess K=2"
  scores this high on its own; re-check once a dataset with real K variance,
  e.g. MuSiQue, is in the loop). Getting here took six real bugs, in the
  order found: (1) `num_of_docs=10` == the whole distractor pool, trivial
  full recall; (2) slot hypotheses were raw interrogative queries, not
  declarative fragments NLI expects; (3) no entity binding (Algorithm 1 line
  4) — an unbound hypothesis like "the actor known for X" can't reject a
  same-topic distractor about the wrong entity; (4) `E5Embedder` never added
  the `'query: '` prefix e5-base-v2 requires, collapsing cosine similarity
  into an uninformative range; (5) `tau_new=0.55` (design doc's placeholder)
  never once cleared past real E5 similarity (min observed 0.660 with the
  prefix fix) — see config.py's `tau_new` comment for the calibrated value
  and the sweep behind it; (6) **the dominant one**: `CrossEncoderNLIScorer`
  read entailment off the wrong output index (`-1` == neutral, not
  entailment, for sentence-transformers' `cross-encoder/nli-*` label
  convention) — this alone explained the AUC being stuck *below* chance
  regardless of every other fix. Moral: when a metric won't move across
  several independently-plausible fixes, suspect the metric's own plumbing
  before the hypothesis. Next: Section 9 Week 2 (VOI gate + GRPO scaffold).
- **Week-2 VOI gate probe: validated** (2026-07-08, `--use_acec` wired into
  `inference_new.py`, 500-sample HotpotQA distractor, calibrated observation
  model): judge 0.79 vs 0.49 baseline, em_processed 0.56-0.57 vs 0.31, *and*
  fewer average docs retrieved (4.0-4.07 vs 5.21) — accuracy gain, not bought
  with more retrieval. Reproduced across two independent 500-sample runs.
  Benefits HotpotQA's `bridge` and `comparison` question types about equally
  (em_processed 0.559 vs 0.573) — expected, since the VOI gate is a general
  "have I covered enough" signal, not bridge-structure-specific.
  **Posterior-driven bridge-entity query rewriting fired 0 times across
  1,000+ samples** (three runs: 20/500/500, one rewrite total, in the very
  first 20-sample check) — the mechanism is implementation-verified (see the
  targeted unit test in the PR) but essentially unexercised at this scale/
  distribution: R3-RAG-Qwen rarely gets stuck re-probing an unresolved slot on
  HotpotQA, and the VOI gate's early stopping likely narrows the window
  further. Not a negative result on the mechanism, just an untested one —
  revisit against a subset selected for repeated-re-probing behavior, or a
  longer-chain dataset (MuSiQue), before concluding either way.
- **Correction to the above, same day**: the "+30pt judge" comparison was
  against a baseline with *no stopping mechanism at all*. The real question —
  does ACEC's VOI gate beat the *old* `BeliefState`'s own stopping heuristic
  (Condition A/B: `llm_reliability > 0.92` OR `ΔQ < 0.05` x2)? Ran that arm,
  same 500-sample setup: old belief system scores judge=0.766, em_processed=
  0.564, avg_docs=4.15 — statistically indistinguishable from ACEC's
  0.79/0.56-0.57/4.05-4.07. **The two belief systems tie**; the earlier win was
  "any reasonable stopping signal beats none," not "ACEC's belief is smarter."
  Exactly the risk the design doc itself flagged (Section C2, Section 8 risk
  #1): frozen-policy inference-time belief interventions are the cheap probe,
  not the main event. ACEC's actual thesis-level advantage — dense per-turn
  coverage reward shaping during GRPO training, which the old system
  structurally cannot provide — remains untested. Next: GRPO scaffold (design
  doc Section 9 D11-14) is now the real test of whether ACEC earns its
  complexity, not another inference-time-only probe.
- **GRPO scaffold, ablation (b) only — correctness check PASSED, no result
  yet** (2026-07-08). Both existing GRPO-RSF training paths
  (`rag/train/grpo_rsf_simple.py`, single-GPU/no-OpenRLHF; and
  `rag/train/R3RAG_OpenRLHF/openrlhf/trainer/ppo_utils/experience_maker_grpo_rsf.py`,
  the distributed path) predate the ACEC design doc by two months and had the
  exact bug Section 3.2 warns about: one scalar GRPO advantage per trajectory,
  broadcast to every turn, which collapses dense per-turn `R_sf` into a
  final-coverage bonus under a shared-question group baseline. Fixed in both
  with `turn_level_returns`/`turn_level_advantages` (ported from
  `acec/reward.py`'s implementation of the same, verified bit-for-bit
  identical — not imported directly, to keep `grpo_rsf_simple.py`
  dependency-light and avoid pulling `acec/__init__.py`'s eager
  torch/sentence-transformers imports into the OpenRLHF training env).
  `grpo_rsf_simple.py` also had no chat template at all (`inference_new.py`
  and the OpenRLHF path wrap every turn in `ApplyChatTemplate`'s
  `<|im_start|>system...user...assistant` — without it R3-RAG-Qwen doesn't
  reliably produce the `Step N:\n` format, so every rollout was hitting the
  format-error branch after turn 1, i.e. `avg_turns` stuck at 1.00, looking
  like "always answers immediately" when it was really "never emits
  parseable output"). Fixed by porting `ApplyChatTemplate` locally. Ran a
  20-sample/2-episode dry run of `grpo_rsf_simple.py` with the retriever
  server *not* running (real-retrieval infra untested on this box, see below)
  — no crash across both episodes, `avg_turns` ticked up to 1.12 after the
  chat-template fix (a couple of rollouts did reach turn 2), loss ≈0 as
  expected when a GRPO group's rewards have ~zero variance (most rollouts hit
  the same format penalty with no working retrieval). This confirms the
  turn-level-advantage code path runs and is exercised on >1-turn
  trajectories, but is **not** a result on ablation (b) itself — that needs a
  working retriever (real `R_sf` signal) and the design's actual smoke-run
  scale (100 steps, 5k prompts, G=8, `20_train_grpo_rsf.sh`). Next: stand up
  `02_start_retriever.sh` on this box (currently hardcoded to `/root/...`,
  needs the same portability fix already applied elsewhere; untested whether
  this box has the RAM/GPU budget for a full wiki18 FAISS index — see
  Environment notes below) before running ablation (b) for a real number.
- **Real retriever stood up (H100 box, real wiki18 index) — hit a second,
  more serious format bug, root-caused and fixed** (2026-07-09). With the
  retriever actually live, avg_turns went *back down* to a flat 1.00 (worse
  than the earlier no-retriever dry run's 1.12) — 100% format-error across
  every rollout at every temperature 0.001–0.9. Root cause: `grpo_rsf_simple.py`
  loads the model with `attn_implementation="eager"`. R3-RAG-Qwen is
  fine-tuned to reproduce an exact literal template (`Step N:\nThe problem
  analysis: ...\nThe retrieval query: ...`), and eager attention's bf16
  numerics diverge from vLLM's (what `inference_new.py` actually uses) just
  enough that, compounding token-by-token over the generation, the model
  reliably stops emitting the literal `The problem analysis:` label — not a
  parsing bug, not a chat-template bug (confirmed the template itself
  matches `tokenizer.apply_chat_template()` almost exactly), a numerics-
  sensitivity issue specific to eager attention on this narrowly-formatted
  model. Confirmed by direct A/B on the identical prompt: vLLM 4/4 correct,
  eager 0/4, `sdpa` 2/2 correct (both greedy and temp=0.7). Fixed by
  switching to `attn_implementation="sdpa"` (built into torch, no flash-attn
  install needed). Moral, same shape as the Week-1 NLI-scorer bug: when a
  metric is stuck at a suspicious constant across every input, suspect the
  inference/numerics plumbing before the higher-level logic. Also found
  (unrelated, while comparing against the OpenRLHF path as a reference
  implementation): `.gitignore`'s unanchored `data`/`models`/`logs` patterns
  matched same-named directories anywhere in the tree, silently swallowing
  `rag/train/R3RAG_OpenRLHF/openrlhf/models/` — never committed, so
  `20_train_grpo_rsf.sh` 404s with `ModuleNotFoundError` on any fresh clone,
  independent of the `peft` dependency issue fixed earlier. Anchored the
  patterns; the missing `openrlhf/models/` content itself still needs
  sourcing from elsewhere before the OpenRLHF path works on a fresh clone.
- **Full wiki18 corpus + E5 index — prebuilt, don't rebuild.** FlashRAG
  (vendored at `rag/tool/FlashRAG/`) publishes a prebuilt e5-base-v2 Flat
  index over wiki18_100w on ModelScope — same filenames `01_build_index.sh`/
  `02_start_retriever.sh` already expect, so this is almost certainly the
  original source those scripts were written against:
  - Index: `https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip`
  - Corpus: `https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w.jsonl`
  Download + unzip into `$HOME/data/indices/e5_Flat/` (verify the extracted
  index filename matches `e5_Flat.index`, or adjust the path) and
  `$HOME/data/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl` — skips
  `01_build_index.sh`'s multi-hour E5 encoding of 17.3M passages entirely, go
  straight to `02_start_retriever.sh`. ModelScope (vs. HuggingFace) is also
  the better bet network-wise from a China-based box, same reasoning as the
  hf-mirror note for model downloads.
- **`openrlhf/models/` recovered from upstream** (2026-07-09): the missing
  directory noted above was never committed anywhere in this repo's history
  (confirmed: not on the H100 box that had a working copy either — that copy
  was untracked local state, not in git). Recovered from
  `github.com/Yuan-Li-FNLP/R3-RAG` (same authors as R3-RAG-Qwen; this whole
  `rag/train/R3RAG_OpenRLHF/` tree was originally vendored from there —
  confirmed by diffing `experience_maker.py` byte-for-byte identical against
  that repo). `20_train_grpo_rsf.sh`'s import chain now gets past
  `ModuleNotFoundError` and fails later, on the actually-still-missing
  `deepspeed` — see next entry for why that path isn't worth chasing further
  right now.
- **`grpo_rsf_vllm.py` added — vLLM-accelerated rollout generation, UNTESTED**
  (2026-07-09, GPU box was off when written). `grpo_rsf_simple.py`'s serial
  one-rollout-at-a-time HF `.generate()` measured at ~17s/rollout — the
  design's actual smoke-run scale (100 steps, 5k prompts, G=8 ≈ 6400
  rollouts) would be ~30h on one GPU. Checked whether `20_train_grpo_rsf.sh`
  (OpenRLHF) would do better and it would not: `GRPORsfExperienceMaker`
  extends `NaiveExperienceMaker`, not the vLLM-capable `RemoteExperienceMaker`
  — it's the same one-at-a-time `Actor.generate()` (confirmed: `actor.py`
  line 147 is a thin wrapper around plain HF `.generate()`), so OpenRLHF
  would add deepspeed/ray overhead with no throughput win on a single GPU.
  OpenRLHF's `Actor` also hardcodes `attn_implementation = "eager"` unless
  `use_flash_attention_2=True` — no `sdpa` option — so it would hit the exact
  format bug above unless flash-attn is actually installed (the
  `USE_FLASH_ATTN=0` skip-it escape hatch in `20_train_grpo_rsf.sh` would
  silently produce the same 100%-format-failure results grpo_rsf_simple.py
  hit before the sdpa fix). `requirements.txt` also pins
  `transformers==4.46.3`/`deepspeed==0.15.0` against the venv's actual
  `transformers 5.13.0` (installed for vLLM/sentence-transformers) — a real
  downgrade/conflict risk, unverified whether deepspeed 0.15.0 even builds
  against torch 2.11.0. Given all that, built `grpo_rsf_vllm.py` instead:
  vLLM (`enable_lora=True`) generates rollouts batched at each turn boundary
  (already confirmed both fast and format-correct for this model, see above
  entries); after every gradient step the updated LoRA adapter is
  hot-swapped into vLLM via a fresh `LoRARequest` (new id + path each step);
  the actual policy/ref log-probs and backward pass still run on a separate
  HF+PEFT model, reusing `grpo_rsf_simple.py`'s `grpo_loss_fn` unchanged.
  vLLM API usage (`LoRARequest`, `LLM(enable_lora=..., max_lora_rank=...)`,
  `SamplingParams.max_tokens`, `RequestOutput.prompt_token_ids` /
  `CompletionOutput.token_ids`) was checked against the actual
  `vllm-project/vllm` source (not from memory) to reduce API-mismatch risk,
  but the script has **not been run** — known risk areas (vLLM's LoRA layer
  coverage for `target_modules="all-linear"`, `max_lora_rank` value
  restrictions, GPU memory headroom for holding both the vLLM engine and the
  separate training model) are flagged in its module docstring. Run the tiny
  correctness pass documented in `20c_train_grpo_rsf_vllm.sh` before trusting
  the full smoke-run-scale defaults.

## Repo layout
- `run_scripts/` — all pipeline entry points, at the repo root (a sibling of
  `rag/`, not nested inside it — every script assumes this; a stray
  `rag/run_scripts/` existed briefly and caused real path bugs before it was
  merged back in here).
- `rag/` — the R3-RAG codebase (benchmark inference/retriever code, the
  `.venv` for vllm, training frameworks under `rag/train/`, vendored tools
  under `rag/tool/`).
- `rag/src/belief/acec/` — see above.
- `rag/legacy/` — archived, not part of the active pipeline (currently: an
  old Gradio demo). See its README before reviving anything there.
- `data/`, `models/`, `logs/` — repo-root siblings of `rag/` (not nested
  inside it), always present (via `.gitkeep`) but gitignored past that, so a
  fresh clone/instance never needs `mkdir` before downloading models/datasets
  or running inference. Matches what `run_scripts/*.sh` actually hardcode:
  `$HOME/data/flashrag_datasets`, `$HOME/models`, `$HOME/logs` (`$HOME` ==
  repo root on the cluster). An earlier version of this scaffold nested
  `data` under `rag/` — wrong, didn't match any script default, removed.
- `experiments/` — small, versioned per-run results (metrics + notes), not
  raw logs. See its README for the convention.
- `run_scripts/.env_retriever` is runtime-generated per machine (by
  `02_start_retriever.sh`/`03_start_split_server.sh`) and gitignored — don't
  expect it to exist on a fresh clone, and don't commit it if it reappears.

## Environment / infra notes worth re-deriving before relying on them
Everything below (conda vs venv split, model/data download steps, GPU/RAM
sizing, path migration) predates ACEC and has not been re-verified against
the current cluster. Treat it as a starting point, not a source of truth —
confirm against the actual machine (e.g. available CPU RAM constrains whether
real-retrieval mode is even feasible; distractor mode does not need a
retriever/FAISS at all).
