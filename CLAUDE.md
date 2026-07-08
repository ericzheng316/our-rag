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
  longer-chain dataset (MuSiQue), before concluding either way. Next: GRPO
  scaffold (design doc Section 9 D11-14).

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
