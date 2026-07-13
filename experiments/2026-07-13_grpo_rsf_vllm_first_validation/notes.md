# grpo_rsf_vllm.py first live validation (2026-07-13)

## What this is
First real run of `rag/train/grpo_rsf_vllm.py` (vLLM-accelerated rollout
generation + LoRA hot-swap + lightweight HF+PEFT gradient step), written
2026-07-09 while the GPU box was off and therefore untested until now.
20-sample/3-episode correctness check, real retriever (wiki18_100w full
index). Single H100 80GB.

Same caveat as the grpo_rsf_simple.py dry run this mirrors: **correctness
check, not an ablation-(b) result.** Scale is far below the design's smoke
run spec (100 steps, 5k prompts, G=8, LoRA r=64).

## Result (second attempt, commit 5944945 — see "what broke" below)
| | Episode 1 | Episode 2 | Episode 3 |
|---|---|---|---|
| avg_turns | 2.72 | 2.88 | 2.38 |
| mean_R | 0.161 | 0.345 | 0.615 |
| loss | -0.0010 | -0.0009 | -0.0043 |

0 connection errors, retriever stayed up the whole run, 140 successful
retrieval requests served. GPU memory: 63.1 GiB / 80 GiB with *both* the
vLLM engine and the separate HF+PEFT training model loaded simultaneously —
comfortable headroom, no OOM.

avg_turns (2.38–2.88) and mean_R (positive throughout, trending up across
the 3 episodes) are in the same range as `grpo_rsf_simple.py`'s already-
validated real-retrieval run (2026-07-09: 2.25–2.81 avg_turns) — confirms
the vLLM-generated rollouts are just as format-correct and are producing
real reward signal, at a fraction of the wall-clock (this run's 3
episodes × 32-ish rollouts each finished in a few minutes, vs.
grpo_rsf_simple.py's serial ~17s/rollout).

mean_R trending up (0.161 → 0.345 → 0.615) across only 3 gradient steps on
a repeated 20-sample pool is *not* strong evidence of real learning (too
few steps, too small/repeated a sample pool — could easily be noise or
early overfitting on 20 fixed questions) — don't over-read it. What it does
confirm: the LoRA hot-swap survived two full swaps without corrupting
anything, and reward isn't degenerate.

## What broke on the first attempt, and the fix
First attempt (commit 17ff6f8, run `grpo_rsf_vllm_20260713_130046`) used
`ThreadPoolExecutor(max_workers=16)` to fire each turn's batch of retrieval
HTTP calls concurrently, on the theory that they're I/O-bound and not GPU
work. That crashed `retrieve_server.py` mid-run:
`BLAS : Program is Terminated. Because you tried to allocate too many
memory regions.` — the retriever runs single-request OpenBLAS-threaded E5
query encoding (`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
MKL_NUM_THREADS=8`) and was never built to take concurrent requests. Worth
noting: the run *did not crash outright* even after the retriever died —
`grpo_rsf_simple.py`'s existing `_retrieve()` exception handling (3
retries, then returns `[]`) kept training going with degraded/empty
retrieval for the rest of that run (still completed 3 episodes, just with
noisier reward). Fixed by reverting to serial retrieval calls (commit
5944945) — retrieval was never the bottleneck this script exists to fix
(GPU generation was), so the concurrency wasn't worth the risk.

## Next
Scale up per design doc D11-14: LoRA r=64, G=8, batch_size=8,
max_samples=5000, num_episodes=100 (`bash run_scripts/20c_train_grpo_rsf_vllm.sh`
with no env overrides — those are already its defaults). Time this run's
wall-clock once started to get a real throughput number vs. the ~30h
serial estimate this script was built to beat.
