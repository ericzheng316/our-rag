# grpo_rsf_vllm.py --use_acec first live validation (2026-07-14)

## What this is
First real run of `--use_acec` (belief-shaped `R_cov` reward, via
`ACECBeliefState`, instead of the gold-SF marginal used in every prior
`grpo_rsf_vllm.py` run). 20-sample/3-episode correctness check, real
retriever (wiki18_100w full index), real calibrated observation model.
Single H100 80GB.

Same caveat as every prior dry run this mirrors: **correctness check, not
an ablation result.** Too small/few-step to say anything about whether
belief-shaped reward beats gold-SF (`experiments/2026-07-13_grpo_rsf_vllm_first_validation/`)
or the frozen-policy inference-time result already on record in CLAUDE.md.

## Prerequisite: recreated the Week-1 calibration artifact
The `observation_model.json` referenced throughout the design doc's Week-1
gate genuinely did not exist on this persistent volume (confirmed via
`find` — both it and its source `records.jsonl` were missing). Reran
`14_run_acec_distractor_pilot.sh` (500 samples) →
`15_build_acec_calibration.sh` (`cross-encoder/nli-deberta-v3-base`):
**GATE PASS**, per-slot hit AUC 0.797 (fit) / 0.875 (held-out), K-accuracy
1.000. Output: `/root/logs/acec_calibration_pilot/observation_model.json`
(persists via the `$HOME/logs -> /root/data/logs` symlink).

Also fixed: `CrossEncoder(model_name)` hits the HF Hub over the network on
*every construction*, even when the model is already fully cached locally
— `HF_HUB_OFFLINE=1` (plus `HF_HOME` pointed at the real persistent cache)
skips that round-trip entirely instead of just rerouting it through
hf-mirror. Wired into `20c_train_grpo_rsf_vllm.sh`'s `USE_ACEC=1` branch.

## Result
| | Episode 1 | Episode 2 | Episode 3 |
|---|---|---|---|
| avg_turns | 2.59 | 2.91 | 2.44 |
| mean_R | 0.057 | 0.010 | 0.390 |
| loss | -0.0092 | 0.0010 | 0.0020 |

0 connection errors, all four models resident on one GPU simultaneously
(vLLM engine, HF+PEFT training model, E5Embedder, NLI cross-encoder):
55.3 GiB / 80 GiB — comfortable headroom, no OOM. avg_turns (2.44–2.91) is
in the same range as the gold-SF run (2.38–2.88), so belief construction/
scoring isn't distorting rollout behavior in some obviously broken way.

mean_R here is smaller in magnitude and non-monotonic (0.057 → 0.010 →
0.390) versus gold-SF's mean_R (0.161 → 0.345 → 0.615, monotonically
increasing) on the same 20-sample pool — expected on priors, since
potential-shaped `R_cov` and the gold-SF marginal are different reward
scales measuring different things, and 3 steps on 20 repeated questions is
too little to read anything into either trend. What this run actually
confirms: the belief reward path is live end-to-end (embedder → NLI
scorer → calibrated observation model → `TurnResult` → potential-shaped
reward → GRPO turn-level advantage → gradient step) without crashing and
without degenerate reward (not stuck at 0, not NaN, not constant).

## Next
Real comparison requires more than 3 steps on 20 fixed questions. Before
scaling to the full design-doc smoke-run spec (LoRA r=64, G=8, 100
episodes, 5k prompts) for `--use_acec`, worth deciding: run gold-SF and
ACEC at the *same* intermediate scale (e.g. a few hundred steps, few
hundred prompts) first to get a real signal on whether belief-shaped
reward changes rollout/learning behavior at all — full smoke-run scale is
expensive to redo if that comparison surfaces something (e.g. reward scale
mismatch) worth fixing first.
