# Experiments

Small, versioned records of what each experiment run actually produced —
metrics and a short write-up, not raw model output. This is what survives
across machines and gets diffed in PRs; it complements (doesn't replace) the
convention documented in `run_scripts/`:

```
$HOME/logs/<run-name>/
  inference.log        # full stdout of the run
  records.jsonl        # per-sample predictions + trajectory (large, stays local)
  results.json         # final metrics dict
  results.csv          # same, as CSV
```

Raw logs above (`records.jsonl` especially, often GBs) belong in
`rag/log/` locally (gitignored — see repo root `.gitignore`) or `$HOME/logs/`
on a cluster box, never in git.

## Convention

One directory per run, named `<YYYY-MM-DD>_<short-name>/`, containing just:

- `results.json` / `results.csv` — copied from the run's `$HOME/logs/<run-name>/`
- `notes.md` — config (dataset, num_search, docs_per_turn, belief on/off,
  git commit hash of the code that produced it), and anything about the
  result worth remembering that isn't obvious from the numbers alone

Example, once the first ACEC calibration pilot has been run:

```
experiments/
  2026-07_acec_week1_calibration_pilot/
    results.json    # from 15_build_acec_calibration.sh's output_dir
    notes.md        # sample count, AUC/K-accuracy gate outcome, NLI model used
```

Nothing is pre-populated here yet — this README exists so the convention is
established before the first result lands, not to speculatively reconstruct
old numbers.
