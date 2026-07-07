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
