# ACEC-v4 selected-evidence calibration, 25 episodes (2026-07-20)

## Bottom line

ACEC-v4 fixed a real calibration-label alignment bug: the supervised label is
now attached to the exact document that produced the runtime max-NLI score,
instead of becoming positive when any other document in the retrieval batch
has a Gold supporting-fact title. The artifact build, held-out calibration
gate, one-episode smoke run, and full two-H100 25-episode run all completed
without an estimator, OOM, or runtime failure.

The corrected v4 calibration did **not** improve the current ACEC training
result. Across 25 episodes, v4 averaged 0.2184 online EM, versus 0.2319 for v3
and 0.2685 for Gold-SF. Relative to v3, v4 made 0.172 fewer retrieval calls
per rollout, but also lost 0.0447 Gold supporting-fact recall and 0.0134 EM.
Relative to Gold, it had nearly the same retrieval cost (-0.034 calls) while
losing 0.0500 EM.

The result should not be read as evidence that the old batch-level label was
correct. It shows that v3's overly optimistic belief updates acted as a useful
retrieval heuristic, while v4 exposed a deeper limitation: a role-aware Platt
calibrator over one max-NLI scalar cannot reliably decide whether an entity is
safe to bind. The next change should improve the selected-evidence target and
verifier, and separately validate the operational binding decision. Lowering
the binding threshold alone would reintroduce false bindings.

## Shared training configuration

- Git commit: `a4d44be1f78194c47a6761106c4cb214965f7e99`
- Model: `/root/data/models/R3-RAG-Qwen`
- Data: HotpotQA `train_sf.jsonl`, 400 sequential questions
- Episodes: 25
- Per episode: 16 questions x 8 samples
- LoRA rank: 64
- Rollout temperature: 0.7, identical for every sample
- One optimizer update per rollout batch
- vLLM GPU memory fraction: 0.60 on H100 GPU 0
- Retriever: full FP32 GPU FAISS on H100 GPU 1, batched search
- GRPO-v2: raw-model policy/reference logprobs, detached pre-step HF policy as
  `pi_old`, non-negative KL estimator, engine/HF MAE gate at 0.05
- ACEC-v4: tau=0.90, fixed K=2, selected-evidence-title label schema
- Reward coefficients: answer 1.0, coverage 0.3, format error 0.1,
  retrieval cost 0.05

All ratio means were 1.0 and all clip fractions were 0, as expected for one
on-policy optimizer update from the exact pre-step snapshot. Average vLLM/HF
raw-logprob MAE was 0.00374.

## V4 artifact and label audit

The artifact was built from 500 logged HotpotQA trajectories and was split by
record into 300 fit, 100 validation, and 100 test records.

- Fit: 691 labeled rows, 118 positive, 573 negative, 39 unknown excluded;
  positive rate 0.1708
- Validation: raw NLI AUC 0.8770, posterior AUC 0.8761, Brier 0.1263,
  ECE 0.1279
- Test: raw NLI AUC 0.8996, posterior AUC 0.8761, Brier 0.1609,
  ECE 0.1615
- The artifact gate passed and fixed-K accuracy was 1.0
- V4 target-action priors: EXPAND 0.1018, REWRITE 0.1646,
  DECOMPOSE 0.1728
- V3 target-action priors were materially higher: 0.1509, 0.2215, 0.2312
- A direct audit found at least 64 old-v3 false positives where the batch
  contained the assigned Gold title but the max-NLI document was a distractor
- 116 corrected negative rows nevertheless had raw NLI >= 0.9, confirming a
  high-score false-positive problem in the verifier/label combination

The AUC improvement is real but insufficient. AUC measures ranking, whereas
the runtime needs a trustworthy high-confidence decision. On corrected
validation/test labels, v4 rows with posterior >= 0.7 had only about
45%/58% precision and about 67%/63% recall. The posterior is therefore not
reliable enough to interpret 0.7 as a 70% binding confidence.

The calibration change also moved the single-update binding boundary sharply:

| Target action | V3 NLI needed for posterior 0.7 | V4 NLI needed for posterior 0.7 |
|---|---:|---:|
| EXPAND | 0.823 | 0.944 |
| REWRITE | 0.778 | 0.909 |
| DECOMPOSE | 0.772 | 0.905 |

This explains the observed behavior: v4 confirms and binds fewer slots, so it
retrieves less, but loses bridge grounding and supporting-fact coverage.

## Aggregate training results

Raw `mean_R` should not be compared between Gold and ACEC because the coverage
term comes from different sources.

| Arm / interval | mean_R | online EM | answer rate | Gold SF recall | retrievals | format error | KL | engine MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4, all 25 | 0.2488 | 0.2184 | 0.9937 | 0.6220 | 1.5416 | 0.0006 | 0.01094 | 0.00374 |
| V3, all 25 | 0.2696 | 0.2319 | 0.9875 | 0.6666 | 1.7132 | 0.0006 | 0.00837 | 0.00372 |
| Gold, all 25 | 0.3589 | 0.2685 | 0.9918 | 0.6429 | 1.5752 | 0.0013 | 0.01285 | 0.00381 |
| V4, first 10 | 0.1695 | 0.1399 | 0.9882 | 0.6449 | 1.6000 | 0.0008 | 0.00158 | 0.00373 |
| V3, first 10 | 0.1866 | 0.1477 | 0.9844 | 0.6702 | 1.6980 | 0.0016 | 0.00168 | 0.00383 |
| Gold, first 10 | 0.2561 | 0.1705 | 0.9852 | 0.6449 | 1.6170 | 0.0008 | 0.00251 | 0.00375 |
| V4, episodes 11-20 | 0.2962 | 0.2625 | 0.9960 | 0.6355 | 1.5100 | 0.0008 | 0.01246 | 0.00374 |
| V3, episodes 11-20 | 0.3171 | 0.2797 | 0.9859 | 0.7007 | 1.7450 | 0.0000 | 0.00958 | 0.00356 |
| Gold, episodes 11-20 | 0.4247 | 0.3210 | 0.9960 | 0.6706 | 1.5530 | 0.0016 | 0.01403 | 0.00383 |
| V4, last 5 | 0.3126 | 0.2874 | 1.0000 | 0.5490 | 1.4880 | 0.0000 | 0.02664 | 0.00376 |
| V3, last 5 | 0.3408 | 0.3046 | 0.9968 | 0.5914 | 1.6800 | 0.0000 | 0.01932 | 0.00382 |
| Gold, last 5 | 0.4328 | 0.3594 | 0.9968 | 0.5836 | 1.5360 | 0.0016 | 0.03116 | 0.00388 |

## Paired episode-index comparison

The runs used the same question ranges and configuration at each episode
index, but vLLM samples were not token-for-token paired. These are diagnostic
single-run comparisons, not a formal multi-seed significance claim.

V4 minus v3:

- Online EM: -0.01344; episode wins/ties/losses 6/5/14
- EM-difference standard deviation 0.03614; standard error 0.00723
- Gold SF recall: -0.04468
- Retrieval calls: -0.17160
- KL: +0.00258

V4 minus Gold:

- Online EM: -0.05004; episode wins/ties/losses 2/4/19
- EM-difference standard deviation 0.05482; standard error 0.01096
- Gold SF recall: -0.02096
- Retrieval calls: -0.03360
- KL: -0.00190

## Runtime and systems checks

- V4 runtime: 51m03s (10:16:36 to 11:07:39 CST)
- V3 runtime: 51m47s
- Gold runtime: 42m54s
- V4 output/checkpoint size: approximately 631 MiB
- GPU 0 peaked below the 81.6 GB H100 limit; no OOM at vLLM fraction 0.60
- GPU 1 held the full FP32 FAISS index at about 64.5 GB
- A 16-query batch-vs-single retriever check returned the same top-5 set for
  all 16 queries and the same order for 15/16. One query swapped ranks 4/5;
  maximum common-document score difference was 0.000506, a harmless FP32 GPU
  numerical tie rather than a retrieval-content change.
- Remote repository was clean and had no `.git/index.lock` after the run

## Interpretation and next decision

1. Keep the v4 argmax score/document alignment contract. Reverting to the old
   batch-any-Gold label would make the probability semantically invalid.
2. Do not launch full-scale GRPO from v4. It is stable and cheaper, but its
   current online accuracy/coverage signal is weaker than v3 and Gold.
3. The next artifact should match slots against the actual HotpotQA supporting
   sentences (title plus sentence text), not an E5 match between a generated
   slot hypothesis and a bare title. Preserve `unknown` for ambiguous cases.
4. Treat coverage estimation and irreversible entity binding as separate
   operational decisions. Add held-out gates for PR-AUC, tail precision/recall,
   and binding accuracy at the deployed threshold; AUC/Brier/ECE alone are not
   enough.
5. Do not simply lower `bound_threshold`: posterior >= 0.7 already has weak
   precision under corrected labels. First improve the verifier/target, then
   tune the threshold on held-out binding utility.
6. After that change, repeat the one-episode smoke and aligned 25-episode pilot.
   Only then run fixed held-out base/Gold/ACEC evaluation and multi-seed,
   full-length GRPO.

## Persistent remote artifacts

- V4 artifact: `/root/data/logs/acec_calibration_v4_20260720/acec_calibration_v4.json`
- Label audit: `/root/data/logs/acec_calibration_v4_20260720/acec_calibration_v4_label_audit.jsonl`
- Artifact metrics: `/root/data/logs/acec_calibration_v4_20260720/acec_calibration_v4_metrics.json`
- One-episode smoke: `/root/data/logs/acec_v4_smoke_1ep_20260720_101017`
- 25-episode train log: `/root/data/logs/acec_v4_25ep_16q8s_fp32_060_20260720/train.log`
- 25-episode checkpoint: `/root/data/logs/acec_v4_25ep_16q8s_fp32_060_20260720/ckpt`

All raw artifacts are on `/root/data`, the instance's persistent disk.
