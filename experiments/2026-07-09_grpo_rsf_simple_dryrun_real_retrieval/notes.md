# GRPO-RSF turn-level-advantage correctness check, real retrieval (2026-07-09)

## What this is
A 20-sample/2-episode dry run of `rag/train/grpo_rsf_simple.py`
(`run_scripts/20b_train_grpo_rsf_simple.sh`), against the **real** wiki18_100w
E5 index (not distractor mode, not an empty/down retriever). Git commit
`0bdeb3b`. Single H100 80GB.

**This is a correctness check on the turn-level GRPO advantage fix (design
doc Section 3.2), not a result on ablation (b) itself.** Scale (20 samples,
2 episodes, G=4) is far below the design's actual smoke-run spec (100 steps,
5k prompts, G=8, via `20_train_grpo_rsf.sh`/OpenRLHF) — don't cite these
numbers as an ablation-(b) EM/judge result.

## Result
| | Episode 1 | Episode 2 |
|---|---|---|
| avg_turns | 2.25 | 2.81 |
| mean_R | 0.156 | 0.150 |
| loss | 0.0012 | -0.0042 |

52 real retriever requests served (checked via retriever.log, all 200 OK).

avg_turns (2.25–2.81) lands in the same ballpark as R3-RAG-Qwen's documented
baseline (mean 1.62 turns, Week-1/2 pilot, CLAUDE.md) rather than the
degenerate 1.00 seen in every earlier attempt — first time this script has
produced real, multi-turn, positively-rewarded trajectories end to end.

## Why earlier attempts all showed avg_turns stuck at 1.00 (and this one didn't)
Three independent bugs had to clear before this run meant anything, in the
order found:
1. Turn-level GRPO advantage was actually trajectory-level (one scalar
   broadcast to every turn) — fixed, ported from `acec/reward.py`.
2. No chat template at all — model never reliably produced the `Step N:\n`
   format — fixed by porting `ApplyChatTemplate`.
3. **The dominant one, found via this run**: `attn_implementation="eager"`.
   R3-RAG-Qwen is fine-tuned to reproduce an exact literal template
   (`Step N:\nThe problem analysis: ...\nThe retrieval query: ...`), and
   eager attention's bf16 numerics diverge from vLLM's (what
   `inference_new.py` actually uses) just enough that, compounding
   token-by-token, the model reliably stopped emitting the literal
   `The problem analysis:` label — 100% format-error rollouts at every
   temperature 0.001–0.9, even with a fully working retriever. Confirmed via
   direct A/B on an identical prompt: vLLM 4/4 correct, eager 0/4, `sdpa`
   2/2 correct. Fixed by switching to `attn_implementation="sdpa"` (no
   flash-attn install needed).

Moral (same shape as the Week-1 NLI-scorer bug): when a metric is stuck at a
suspicious constant across every input/condition, suspect the inference/
numerics plumbing before the higher-level logic.

## Next
Scale this up per design doc D11-14: LoRA r=64, G=8, 100 steps, 5k prompts,
via the OpenRLHF path (`20_train_grpo_rsf.sh`) — that path currently has a
separate, unrelated blocker (`rag/train/R3RAG_OpenRLHF/openrlhf/models/`
missing from git entirely, see CLAUDE.md 2026-07-09 entry) that needs
resolving first.
