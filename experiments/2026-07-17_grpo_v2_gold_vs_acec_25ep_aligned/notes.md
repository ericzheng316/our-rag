# Aligned GRPO-v2: Gold-SF vs ACEC-v3, 25 episodes (2026-07-17)

## Bottom line

The estimator/alignment repair passed the real two-H100 test. Both arms ran
all 25 episodes without the late-stage policy collapse seen in the earlier
Gold run: the policy ratio stayed exactly 1.0 for the single on-policy update,
clip fraction stayed 0, vLLM/HF raw-logprob MAE stayed around 0.004, and format
errors remained negligible.

ACEC-v3 did **not** beat Gold-SF on training-rollout EM in this single run.
Across the same 25 episode indices, ACEC averaged 0.2319 EM versus Gold's
0.2685 (-0.0366 absolute). ACEC won 3 episode-level EM comparisons, tied 2,
and lost 20. It retrieved more often (+0.138 calls per rollout) and achieved
slightly higher Gold supporting-fact recall (+0.0237), but converted that
extra retrieval/coverage into fewer exact-match answers.

This is evidence that ACEC is viable and stable, but the current v3 reward is
not yet evidence of an improvement over Gold supervision. The next decision
must use a fixed held-out evaluation of the base, Gold, and ACEC checkpoints;
training-rollout metrics use different samples each episode and are not a
substitute for evaluation.

## Shared configuration

- Git commit: `4a4999b45950e105ccf9318df731d72140841866`
- Model: `/root/data/models/R3-RAG-Qwen`
- Data: HotpotQA `train_sf.jsonl`, 400 sequential questions
- Episodes: 25
- Per episode: 16 questions x 8 samples
- LoRA rank: 64
- Rollout temperature: 0.7, identical for every sample
- One optimizer update per rollout batch
- vLLM GPU memory fraction: 0.60 on H100 GPU 0
- Retriever: full FP32 GPU FAISS on H100 GPU 1, batched search
- Unified reward coefficients: answer 1.0, coverage 0.3, format error 0.1,
  retrieval cost 0.05
- GRPO-v2: raw-model policy/reference logprobs, detached pre-step HF policy as
  `pi_old`, non-negative KL estimator, engine/HF per-turn MAE gate at 0.05
- ACEC-v3: tau=0.90 calibration artifact, fixed K=2
- No estimator, calibration, OOM, or runtime failure occurred in either run.

## Aggregate results

Raw `mean_R` is included within each arm but should not be compared directly
between arms: Gold and ACEC use different sources for the coverage term.

| Arm / interval | mean_R | online EM | answer rate | Gold SF recall | retrievals | format error | KL | engine MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gold, all 25 | 0.3589 | 0.2685 | 0.9918 | 0.6429 | 1.5752 | 0.0013 | 0.01285 | 0.00381 |
| ACEC, all 25 | 0.2696 | 0.2319 | 0.9875 | 0.6666 | 1.7132 | 0.0006 | 0.00837 | 0.00372 |
| Gold, first 10 | 0.2561 | 0.1705 | 0.9852 | 0.6449 | 1.6170 | 0.0008 | 0.00251 | 0.00375 |
| ACEC, first 10 | 0.1866 | 0.1477 | 0.9844 | 0.6702 | 1.6980 | 0.0016 | 0.00168 | 0.00383 |
| Gold, episodes 11-20 | 0.4247 | 0.3210 | 0.9960 | 0.6706 | 1.5530 | 0.0016 | 0.01403 | 0.00383 |
| ACEC, episodes 11-20 | 0.3171 | 0.2797 | 0.9859 | 0.7007 | 1.7450 | 0.0000 | 0.00958 | 0.00356 |
| Gold, last 5 | 0.4328 | 0.3594 | 0.9968 | 0.5836 | 1.5360 | 0.0016 | 0.03116 | 0.00388 |
| ACEC, last 5 | 0.3408 | 0.3046 | 0.9968 | 0.5914 | 1.6800 | 0.0000 | 0.01932 | 0.00382 |

All ratio means were 1.0000 and all clip fractions were 0.0000, as expected
for one optimizer update from the exact pre-step policy snapshot.

## Paired episode-index comparison

The two arms used the same question ranges at each episode index, but vLLM
sampling was not token-for-token paired, so these numbers describe this run
and are not a formal significance test.

- ACEC minus Gold online EM: -0.0366
- Standard deviation of episode-index EM differences: 0.0468
- Standard error of the mean difference: 0.0094
- ACEC episode wins / ties / losses: 3 / 2 / 20
- ACEC minus Gold SF recall: +0.0237
- ACEC minus Gold retrieval calls: +0.1380

The observed tradeoff is therefore: ACEC retrieved more and recovered a bit
more Gold evidence, while Gold answered more questions exactly.

## Runtime

- ACEC: 51m47s (10:11:32 to 11:03:19 CST)
- Gold: 42m54s (11:18:09 to 12:01:03 CST)
- ACEC overhead: 8m53s, approximately 20.7% over Gold

The extra cost is consistent with ACEC's coverage/NLI work and its higher
average retrieval count. This overhead occurred offline in the training run;
the recommended held-out evaluation does not need to run on every optimizer
step.

## Interpretation and next decision

1. Do not change the estimator again based on these results. Its stability and
   alignment checks passed.
2. Evaluate the base model and both final adapters on one fixed held-out set.
   Report EM, token F1, Gold SF recall, retrieval calls, answer rate, and format
   errors. Reuse exactly the same questions and evaluation settings.
3. Add a post-hoc diagnostic comparing ACEC coverage/delta coverage with Gold
   SF recall per trajectory. The current result suggests extra ACEC-driven
   retrieval is not being converted efficiently into exact answers.
4. If the held-out result confirms the online EM gap, work next on ACEC labels,
   calibration, and K selection (especially fixed K=2), not GRPO alignment.
5. Only after the held-out protocol is working should the comparison be
   repeated with multiple seeds.

## Persistent remote artifacts

ACEC:

- Launch log: `/root/data/logs/acec_v3_25ep_16q8s_fp32_060_20260717.launch.log`
- Train log: `/root/data/logs/acec_v3_25ep_16q8s_fp32_060_20260717/train.log`
- Checkpoint: `/root/data/logs/acec_v3_25ep_16q8s_fp32_060_20260717/ckpt`

Gold:

- Launch log: `/root/data/logs/gold_corrected_25ep_16q8s_fp32_060_20260717.launch.log`
- Train log: `/root/data/logs/gold_corrected_25ep_16q8s_fp32_060_20260717/vllm_0.60/train.log`
- Checkpoint: `/root/data/logs/gold_corrected_25ep_16q8s_fp32_060_20260717/vllm_0.60/ckpt`

The raw artifacts are on `/root/data`, the instance's persistent disk.
