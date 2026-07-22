# Outcome-only GRPO ablation

## Decision this experiment supports

Does ACEC's process signal improve fixed held-out answer quality and evidence
coverage beyond the improvement obtained by answer-outcome-only GRPO under the
same model, data order, group size, temperature, optimizer, KL, and compute?

## Arms

| Arm | Terminal EM | Coverage reward | Retrieval cost | Format penalty | ACEC runtime |
| --- | ---: | ---: | ---: | ---: | --- |
| Outcome-only | 1.0 | 0.0 | 0.0 | 0.0 | No |
| ACEC-v5 | 1.0 | 0.3 × marginal coverage | 0.05/retrieval | 0.1 | Yes |

Outcome-only deliberately excludes gold supporting-fact reward. A gold-SF arm
would answer a different question: whether ACEC can replace privileged process
labels, not whether process shaping itself adds value over terminal outcome.

## Fixed controls

- Base model: R3-RAG-Qwen.
- HotpotQA training order: first 1,600 records, sequential 16-question episodes.
- Eight rollouts per question, one shared temperature 0.7 per GRPO group.
- LoRA rank 64, learning rate 5e-5, KL coefficient 0.01, clip epsilon 0.2.
- Checkpoints at 25/50/75/100; primary comparison uses 50 and 100.
- Held-out: the same 256 HotpotQA dev ids, seed 20260721, temperature 0.

## Metrics and decision rule

Primary outcome: fixed held-out processed EM and processed F1.

Drivers: direct EM/F1 and semantic answer accuracy. Guardrails: Gold-SF recall,
retrieval calls per question, answer rate, and format-error rate.

Evidence for ACEC requires a paired advantage over outcome-only on answer
quality, not merely higher training reward. Retrieval efficiency is a benefit
only when answer quality is non-inferior. Report paired wins/losses and bootstrap
intervals; do not use one online training window as the decision metric.

Because this is one training seed per arm, a small difference is directional.
A paper-level attribution claim should repeat the matched run with at least two
additional seeds or confirm the ordering on a larger fixed held-out set.
