# ACEC-v5 100-episode learning-trend run (2026-07-21)

## Bottom line

The first 100-episode ACEC-v5 GRPO run completed successfully on two H100s.
It processed 1,600 sequential HotpotQA training questions with eight rollouts
per question (12,800 trajectories), saved complete checkpoints at episodes
25/50/75/100, and encountered no OOM, traceback, policy-alignment failure, or
empty answer. The run took 3h42m06s.

The online learning trend was strong: the training answer metric rose from
0.1618 over the first 10 episodes to 0.5415 over the last 10, while Gold
supporting-fact recall rose from 0.6655 to 0.7152 and retrieval calls fell from
1.662 to 1.253 per rollout. KL increased from 0.0018 to 0.0961 over the same
interval, which is material but not by itself abnormal for this length of run.
The final policy remained operationally stable: answer rate was 0.9972 and
only about seven format errors occurred in 12,800 rollouts.

A fixed 256-question HotpotQA dev evaluation was then run for the frozen R3
model and episodes 25/50/75/100. Episode 50 had the best raw whole-answer EM,
raw F1, and Gold supporting-fact recall tradeoff; episode 100 used the fewest
retrieval calls. However, these raw metrics are **not** the R3-RAG paper's
reported EM. R3 first uses an evaluation LLM to extract short candidate
answers and then computes normalized exact match (`em_processed`). Our raw EM
instead compares the complete generated answer, while substring EM merely
checks whether a Gold answer occurs inside it.

Consequently, the held-out run establishes better canonical short-answer
output and a genuine retrieval-side improvement at episode 50, but it does
not yet establish higher semantic answer accuracy. Raw EM rose from 0.0938 to
0.3672 at episode 50, while substring EM fell from 0.4805 to 0.4063. The next
answer-quality evaluation must apply the official R3 processed EM/F1 protocol
and an independent semantic judge to these already-saved predictions.

## Training configuration

- ACEC-v5 runtime commit: `9d31b7c`
- 100-episode runner commit: `eab4961`
- Model: `/root/data/models/R3-RAG-Qwen`
- Data: HotpotQA `train_sf.jsonl`, first 1,600 sequential questions
- Episodes: 100
- Per episode: 16 questions x 8 samples
- Total rollouts: 12,800
- LoRA rank: 64
- Rollout temperature: 0.7, identical across samples
- vLLM GPU memory fraction: 0.60 on H100 GPU 0
- Retriever: full FP32 GPU FAISS on H100 GPU 1, batched search
- Calibration artifact:
  `/root/data/logs/acec_v5_tau075_calib1000_b738c85/acec_calibration_v5.json`
- Reward coefficients: answer 1.0, coverage 0.3, format error 0.1,
  retrieval cost 0.05
- Checkpoint interval: 25 episodes

## Aggregate training results

`mean_R` is meaningful within this ACEC-v5 run. It must not be compared
directly with a Gold-SF reward because the coverage sources differ. The ACEC
coverage values below are derived from the logged reward equation rather than
an independent Gold label.

| Interval | mean_R | online answer metric | Gold SF recall | retrievals | turns | KL | answer rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 100 | 0.5072 | 0.4167 | 0.6743 | 1.4368 | 2.4349 | 0.0568 | 0.9972 |
| First 10 | - | 0.1618 | 0.6655 | 1.6620 | 2.6520 | 0.0018 | - |
| Last 20 | - | 0.5153 | 0.6970 | 1.2875 | - | 0.0910 | - |
| Last 10 | - | 0.5415 | 0.7152 | 1.2530 | 2.2530 | 0.0961 | - |

Additional diagnostics:

- Derived mean ACEC coverage over all episodes: 0.5413
- Derived ACEC coverage per retrieval: 0.3768 overall, 0.3058 for the first
  10 episodes, and 0.4355 for the last 10
- Best single online episode: episode 92, answer metric 0.711, Gold SF recall
  0.875, 1.41 retrieval calls, KL 0.0589
- Format error rate: approximately 0.0006 (about 7/12,800)
- Empty answers: 0

## ACEC runtime behavior

The final runtime summary aggregated 31,156 turns and 19,973 retrieval
observations:

| Action | Count | Share among retrieval actions |
|---|---:|---:|
| DECOMPOSE | 14,474 | 78.7% |
| REWRITE | 2,765 | 15.0% |
| EXPAND | 1,152 | 6.3% |
| ANSWER | 12,765 | - |

- Bindings: 14,354
- Mean delta coverage per retrieval observation: 0.2227
- Mean gain posterior: 0.7292
- Mean novelty: 0.9082
- Mean support: 0.4141
- Exact-repeat fraction: 0
- Low-novelty fraction: 0.00365

The DECOMPOSE concentration is a design risk to audit. It may represent a
useful HotpotQA multi-hop strategy, but it could also reflect a calibrator or
reward prior that over-favors one action. This run alone cannot distinguish
those explanations.

## Fixed held-out evaluation

Evaluation implementation commit: `7e1b03e`. The evaluator sampled the same
256 HotpotQA dev questions for every variant with seed 20260721, temperature
0, one sample per question, top-5 retrieval, at most five iterations, and the
same real retriever and rollout path. ACEC calibration/NLI was not loaded at
inference.

| Variant | Raw whole-answer EM | Raw F1 | Substring EM | Gold SF recall | Retrieval calls | Answer rate |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R3 | 0.0938 | 0.2266 | 0.4805 | 0.6992 | 1.7891 | 0.9883 |
| Episode 25 | 0.2773 | 0.4041 | 0.4219 | 0.7324 | 1.5742 | 1.0000 |
| Episode 50 | **0.3672** | **0.4773** | 0.4063 | **0.7324** | 1.4609 | 1.0000 |
| Episode 75 | 0.3320 | 0.4562 | 0.3828 | 0.7012 | 1.4180 | 1.0000 |
| Episode 100 | 0.3477 | 0.4498 | 0.3828 | 0.6816 | **1.2969** | 1.0000 |

Relative to frozen R3, episode 50 changed:

- Raw whole-answer EM: +0.2734
- Raw F1: +0.2507
- Substring EM: -0.0742
- Gold SF recall: +0.0332
- Retrieval calls: -0.3281 (-18.3%)

The raw EM gain is strongly confounded by canonicalization: R3 often emits a
correct short answer inside a longer explanatory sentence, while the trained
adapters learned to emit a Gold-like short string. Substring EM is also noisy
and is not a semantic judge, but its decline is sufficient to prevent a claim
that answer correctness improved.

## Frozen-R3 loading audit

A separate 16-question audit completely disabled vLLM LoRA support rather
than enabling LoRA with no adapter request. It reproduced the original frozen
baseline aggregates: raw EM 0.0625, substring EM 0.5000, Gold SF recall
0.65625, and 1.75 retrieval calls. EM agreed on all 16 examples and generated
text agreed exactly on 15/16; the remaining answer differed only because of
generation-level numerical nondeterminism and had the same EM outcome.

Both `/root/models/R3-RAG-Qwen` and `/root/data/models/R3-RAG-Qwen` resolved
to the persistent model directory. This rules out accidental application of
an ACEC adapter as the explanation for the low raw baseline EM.

## Metric interpretation and next decision

1. Preserve all four checkpoints. Episode 50 is the current answer/coverage
   candidate; episode 100 is the retrieval-efficiency candidate.
2. Do not report 9.38% to 36.72% as a semantic EM improvement. It is a raw
   full-response exact-match result and contains a large formatting effect.
3. Do not compare substring EM with the R3 paper. The paper's Qwen2.5-7B + E5
   HotpotQA result (EM 46.4, F1 59.7, ACC 65.5) uses short-answer extraction
   before EM/F1 and a separate Qwen2.5-72B semantic judge for ACC.
4. Re-evaluate the saved 5 x 256 predictions with the official R3 candidate-
   answer extraction and normalized processed EM/F1. Add an independent
   semantic judge and audit extraction failures/disagreement cases.
5. Only after that evaluation should the project claim higher answer
   accuracy or select episode 50 over episode 100 for subsequent comparisons.
6. Retrieval-side evidence is already positive at episode 50: Gold SF recall
   increased by 0.0332 while retrieval calls decreased by 18.3%. This remains
   a single fixed subset and is not yet a multi-seed significance result.

## Persistent remote artifacts

- Training root:
  `/root/data/logs/acec_v5_100epi_16x8_9d31b7c`
- Training log:
  `/root/data/logs/acec_v5_100epi_16x8_9d31b7c/train.log`
- Checkpoints:
  `/root/data/logs/acec_v5_100epi_16x8_9d31b7c/ckpt/step_{25,50,75,100}`
- Held-out results:
  `/root/data/logs/acec_v5_heldout_dev256_7e1b03e/results.json`
- Held-out per-variant predictions:
  `/root/data/logs/acec_v5_heldout_dev256_7e1b03e/predictions_*.jsonl`
- No-LoRA frozen audit:
  `/root/data/logs/r3_frozen_no_lora_audit16_7e1b03e`

The remote instance was shut down after the artifacts were written to the
`/root/data` persistent disk.
