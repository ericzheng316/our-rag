# ACEC-v5 R3 processed-EM evaluation (2026-07-22)

## Bottom line

The saved frozen-R3 and ACEC-v5 checkpoint answers were evaluated on the same
256-question HotpotQA dev subset with the R3-RAG candidate-answer extraction
protocol. Qwen2.5-72B-Instruct extracted short candidates at temperature 0,
after which normalized EM/F1 was computed exactly as in the mirrored R3
metric implementation.

Episode 50 remains the best ACEC checkpoint, but its processed answer accuracy
is statistically indistinguishable from frozen R3. Processed EM was 0.3945
for episode 50 versus 0.3867 for frozen R3: a paired increase of only 0.0078
(two net questions), with a 95% bootstrap interval of [-0.0352, 0.0508].
Processed F1 was 0.5008 versus 0.5192, a paired decrease of 0.0184 with a 95%
interval of [-0.0549, 0.0179].

This changes the interpretation of the earlier raw-EM result. Episode 50's
raw whole-answer EM gain from 0.0938 to 0.3672 mostly reflects learning to emit
short, canonical Gold-like answers. After the R3 extractor removes that format
confound, episode 50 is approximately tied with the frozen model rather than
substantially better in extraction-normalized answer correctness. This does not show that
ACEC is useless: the earlier evaluation still found higher Gold supporting-
fact recall with fewer retrieval calls at episode 50. It does show that the
current evidence is insufficient for a paper claim that ACEC significantly
improves processed final-answer EM/F1. A semantic ACC claim still requires the
independent judge.

## Evaluation configuration

- Evaluator commit: `55bb67826965efd243b5e6fc8c013124441b8cd9`
- Population: fixed 256-question HotpotQA dev subset, identical for all models
- Prediction source commit: `7e1b03e`
- Baseline: frozen R3-RAG-Qwen
- ACEC checkpoints: episodes 25, 50, 75, and 100
- Extractor: `Qwen2.5-72B-Instruct`
- Extractor temperature: 0
- Candidate output cap: 512 tokens
- Metric source: R3-RAG `benchmark/R3-RAG/src/cal_metric.py`
- Paired confidence intervals: 10,000 bootstrap samples, seed 20260722
- Initial API jobs: 914; raw-exact/no-answer short circuits: 366
- Successful API extractions: 914; final extraction failures: 0
- API-reported usage: 524,698 input + 12,477 output = 537,175 tokens

## Main results

| Variant | Raw/direct EM | R3 processed EM | R3 processed F1 | Processed exact count |
|---|---:|---:|---:|---:|
| Frozen R3 | 0.0938 | 0.3867 | **0.5192** | 99/256 |
| Episode 25 | 0.2773 | 0.3750 | 0.4963 | 96/256 |
| Episode 50 | **0.3672** | **0.3945** | 0.5008 | 101/256 |
| Episode 75 | 0.3320 | 0.3555 | 0.4708 | 91/256 |
| Episode 100 | 0.3477 | 0.3633 | 0.4681 | 93/256 |

The extractor rescued 75 additional exact answers for frozen R3 (24 direct
to 99 processed) but only seven for episode 50 (94 direct to 101 processed).
That asymmetry is the concrete reason the raw 27.34-point EM gain collapses to
a 0.78-point processed-EM difference.

## Paired comparison against frozen R3

| Variant | Processed-EM delta | 95% CI | Wins / losses | Processed-F1 delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| Episode 25 | -0.0117 | [-0.0586, 0.0313] | 16 / 19 | -0.0229 | [-0.0642, 0.0188] |
| Episode 50 | +0.0078 | [-0.0352, 0.0508] | 16 / 14 | -0.0184 | [-0.0549, 0.0179] |
| Episode 75 | -0.0313 | [-0.0820, 0.0195] | 17 / 25 | -0.0484 | [-0.0939, -0.0033] |
| Episode 100 | -0.0234 | [-0.0742, 0.0273] | 18 / 24 | -0.0511 | [-0.0980, -0.0044] |

No ACEC checkpoint has a statistically resolved processed-EM advantage on
this subset. Episodes 75 and 100 have statistically resolved processed-F1
decreases, which is evidence of answer-quality over-optimization after the
episode-50 region even though raw answer formatting remains strong.

## Validation and recovery audit

The first launch used eight workers and triggered provider rate limits. It
was stopped after 49 records exhausted their retries with `RateLimitError`.
Those failed fallback records were removed after preserving the original
progress/log files, and the same output was resumed with two workers and two
explicit retries. All 616 remaining jobs completed without a failure.

The final artifact was independently checked for:

- 1,280 total variant-question rows and 1,280 unique `(variant, id)` keys
- exactly 256 unique questions in every variant
- 914 extracted, 363 raw-exact-short-circuit, and 3 no-answer records
- zero extraction-failure fallbacks
- independently recomputed per-variant EM/F1 equal to `results.json`
- total per-variant API usage equal to 537,175 tokens

Confidence assessment: **ready to share for the narrow claim that ACEC-v5
episode 50 is at parity with frozen R3 on processed EM on this fixed subset;
share with caveats for broader model-quality or algorithm-generalization
claims.**

## Interpretation and next decision

1. Use episode 50 as the only ACEC checkpoint for the next diagnostic round;
   additional training to episodes 75/100 did not improve processed EM and
   materially reduced processed F1.
2. Audit the paired episode-50 cases: 16 processed-EM wins, 14 losses, and
   cases where Gold-SF recall improves but the final answer does not. This is
   the shortest path to distinguishing retrieval improvement from generator
   conversion failure or reward misspecification.
3. Run the R3-style independent ACC semantic judge on the saved predictions.
   Processed EM is much fairer than raw EM but still depends on an extractor
   and exact normalization.
4. For the paper, report metric names precisely: HotpotQA raw/direct EM and
   R3 processed EM are different protocols. Compare ACEC and baselines under
   the same protocol; do not compare one against a number reported under the
   other.
5. Do not start a larger GRPO run solely on the raw-EM trend. First establish
   whether ACEC improves retrieval evidence in the paired wins/losses and why
   that evidence is not consistently converted into correct final answers.

## Persistent artifacts

- CPU evaluation root:
  `/root/data/logs/acec_v5_r3_processed_em_20260722_103109`
- Final result:
  `/root/data/logs/acec_v5_r3_processed_em_20260722_103109/results.json`
- Final per-example records:
  `/root/data/logs/acec_v5_r3_processed_em_20260722_103109/processed_predictions_*.jsonl`
- Rate-limit recovery backups:
  `/root/data/logs/acec_v5_r3_processed_em_20260722_103109/*workers8*`
- Repository copy: `experiments/2026-07-22_acec_v5_r3_processed_em/results.json`
