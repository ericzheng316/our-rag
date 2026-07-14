# gold-SF vs ACEC, 25-episode head-to-head at real G=8/lora_rank=64 (2026-07-14)

## What this is
First head-to-head at the actual design-doc G=8/lora_rank=64 scale (not the
earlier tiny G=4/rank=16 correctness passes). 25 episodes, 200 distinct
questions (no repeats), 8q x 8samples/episode, single H100. gold-SF ran
first, then ACEC, sequentially on the same GPU, same question order (script
has no shuffling, so both arms saw identical data per episode).

**This is preliminary data, not a conclusion.** 25 episodes is a fraction of
even the design-doc smoke run (100 episodes), let alone a full epoch (1000
episodes for the real 8k-prompt spec). Single run per arm, no repeats to
characterize noise. Treat every number below as "what we saw once," not
"what either reward mode does."

## Headline numbers
| | gold-SF | ACEC |
|---|---|---|
| mean_R avg (25 ep) | 0.419 | 0.166 |
| mean_R std | 0.139 | 0.089 |
| relative variability (std/mean) | 0.33 | 0.54 |
| first-12 -> last-13 ep avg | 0.360 -> 0.474 (+32%) | 0.131 -> 0.199 (+52%) |
| avg_turns avg | 2.593 | 2.456 |
| wall clock | 2h11m | 2h02m |

## Reading these numbers (with real caveats)
- **gold-SF's raw mean_R is ~2.5x ACEC's — not a like-for-like comparison.**
  gold-SF's reward is an SF-title-hit marginal (LAMBDA_SF=0.5 per hit) +
  EM (LAMBDA_ANS=1.0). ACEC's is potential-shaped coverage
  (`eta*(C_after-C_before)`, eta=0.3, so coverage alone caps at +0.3 per
  trajectory) + EM, minus an efficiency penalty. The two live on different
  scales by construction; a 2.5x gap in raw mean_R says nothing about which
  policy is more accurate without also measuring actual EM/judge score,
  which this run did not do (no eval pass, just the training reward log).
- **ACEC is noisier relative to its own mean** (0.54 vs 0.33) — visually
  this is the "反复横跳" (bouncing back and forth) pattern, real and not
  an artifact. Plausible causes, not yet distinguished: (a) ACEC's reward
  is a product of more moving parts (NLI cross-encoder scores -> belief
  update -> calibrated observation model -> potential difference), each
  adding its own noise; (b) small per-episode batch (8 questions, no
  repeats across episodes so each episode is a genuinely different sample
  of HotpotQA difficulty); (c) genuinely higher training instability under
  this reward shape. Not resolved here.
- **ACEC's relative improvement (+52%) is larger than gold-SF's (+32%)**
  despite the lower absolute floor — i.e. by the crude "first-half vs
  second-half" measure, ACEC is not learning slower, just starting and
  staying at a lower reward *scale*. Interesting, but 25 episodes / 2
  twelve-ish-episode halves is nowhere near enough to call this a real
  trend vs noise.
- **avg_turns is close between arms** (2.59 vs 2.46) — ACEC is not
  obviously farming extra turns or truncating early relative to gold-SF.
- **Wall clock: ACEC finished ~9 min faster than gold-SF** despite loading
  two extra models (E5Embedder, NLI cross-encoder) every turn — most
  likely just because ACEC's avg_turns was slightly lower (fewer retrieval
  round-trips), not because ACEC's own reward computation is cheap. Given
  retrieval was serial and CPU-bound this run (batched-retrieval fix,
  commit f51b942, landed *after* both these runs), turn count is probably
  the dominant lever on wall clock, not reward-computation cost.

## What this run does NOT tell us
- Not whether either arm's *policy* actually got better at answering
  questions (no EM/judge eval pass was run against the resulting
  checkpoints — this is training-reward-log data only).
- Not whether ACEC's lower/noisier reward is a real weakness of the
  belief-shaped reward or just this run's specific 200-question sample.
- Not a throughput number under the batched-retrieval fix — both runs here
  predate it (see run configs above).

## Next
- This run's own retrieval was still serial/CPU-bound. The very next
  thing to measure (cheap, no new hardware) is the same kind of run with
  the batched-retrieval fix (commit f51b942) actually in effect, to get a
  real before/after wall-clock number — separate from the reward-quality
  question above.
- A second GPU dedicated to retrieval (GPU-FAISS via `GPU_ID`/`FAISS_GPU`
  in `02_start_retriever.sh`, commit 050e364) is planned once available;
  not yet tested on real hardware.
- Before trusting either arm's reward trend, would want: (a) more episodes
  per arm to separate signal from the noise characterized above, (b) an
  actual EM/judge eval pass on the resulting checkpoints, not just the
  training-time reward log.
