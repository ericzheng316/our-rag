# ACEC v8 — Answer-Anchored Action Lookahead

Status: chartered 2026-08-05, after the v6.4 falsification program concluded.
Design owner: boyuz5. This document records the decisions; the evidence that
forced each one is in Section 6.

## 0. What v8 is

A rebuild of the belief around the only lever that survived falsification —
**action selection over retrieval** — with every learned component supervised
by one terminal objective: **did the final answer come out correct**. All
proxy objectives (slot-hit coverage, NLI grounding, adaptive-stop thresholds)
are demoted or deleted; test-time compute is spent at the decision points that
were measured to carry value.

## 1. Components

1. **Slot-coverage posteriors (kept).** Per-slot hit probabilities with
   Bayesian updates — the one v5 component whose upstream quality was healthy
   (per-slot hit AUC 0.875). Recalibrated against answer outcomes, not
   slot-hit labels.
2. **Slot initialization by a strong instruction-following model (new).**
   Slots/bridge entities are generated offline by Qwen3.5-9B (local), not by
   the rigid R3 policy (0.59% parse) or question heuristics (binding 20.3%).
3. **One unified verifier (new), three roles.** Qwen3.5-9B judges
   (question, evidence, candidate answer) → p(correct). The same module is
   (a) the retrieval reranker — "retrieved but unanswerable" filter,
   (b) the reward model R(s′) for training, and (c) the lookahead judge at
   inference. NLI cross-encoder and the coverage potential are retired.
4. **Three entity-directed actions.** Replacing EXPAND/REWRITE/DECOMPOSE
   (never showed terminal value): `{query current target-slot entity,
   pivot to next-slot entity, answer now}` — the semantics where the +3–7pp
   oracle ceiling lives.
5. **Inference-time expectation over actions (owner's call).** At each
   decision point, branch all three actions, execute retrieval for real,
   score the reached states with the verifier, take argmax. Under greedy
   decoding and deterministic retrieval, one probe per action *is* the
   expectation; with sampling, n=2–3. Depth is a budget knob (depth-1 on
   HotpotQA; deeper on long-horizon sets). Cost ≈ 3× per decision point.
6. **Action prior updated per training round.** A learned P(action | state
   features) fitted on branched-rollout outcomes; serves as the low-budget
   fallback policy and the exploration prior. Exploration is ε-greedy/UCB —
   pure greedy does not explore; with 3 arms exploration is cheap.
7. **Policy stays frozen (R3-RAG-Qwen) during v8 validation.** Baselines and
   the measurement machinery (raw-answer propagation, processed EM) carry
   over unchanged. Retraining enters only at the v8 equivalent of Gate 5.

## 2. Pre-registered offline doors (pass before any pipeline integration)

| # | Door | Testbed (already on disk) | Bar |
|---|------|---------------------------|-----|
| 1 | Slot-init quality | 3,149 audited bridge entities | gold-entity containment ≫ 20.3% baseline |
| 2 | Verifier | 6,144 labeled candidates, question-disjoint splits identical to the failed calibrated verifier (val AUC 0.4883) | val AUC ≥ 0.55 **and** top-1 oracle recovery ≥ 0.50 |
| 3 | Reward-proxy sanity | a0 paired arms (gold vs none) | verifier-based R must prefer gold-arm states on questions where gold won |
| 4 | Horizon / arena | retrieval-calls distribution per dataset | HotpotQA ≈ 1–2 step regime (measured 1.4); long-horizon claims validated on MuSiQue only |

Ideas that fail a door die offline in minutes; nothing failing a door gets
GPU-trained or API-evaluated.

## 3. The budget-matched pre-registration

Lookahead spends ~3–4× inference compute, so its headline comparison is
**against unstructured best-of-N sampling with the same verifier at the same
generation budget**. Only a win here demonstrates that the *action structure*
— not raw test-time compute — carries the value. Sampling ceilings for the
comparison already exist (oracle@2/4/8 per policy).

## 4. Arena

- Offline doors and verifier work: HotpotQA assets (labels exist).
- Action/horizon validation and any training: **MuSiQue** (2–4 hop,
  compositional, unanswerable subset available; downloaded). Every MuSiQue
  evaluation carries a **closed-book control arm** — no-retrieval answers —
  so shortcutable questions are measured out of the action story.
- HotpotQA remains for baseline comparability with the v5–v6.4 record.

## 5. Deleted from the v5–v7 design

- κ (K-posterior): ran as constant K=2 in production; delete.
- Coverage aggregate C_t as reward potential / stop signal: AUC 0.49 against
  correctness, dynamic range std 0.03; delete as objective, keep only as a
  candidate *feature* for the action prior.
- Adaptive-stop thresholds (τ-dial): behaviorally inert; delete.
- monitor-preseed via policy slot generation: 0.59% parse; replaced by door 1.
- NLI observation model: replaced by the unified verifier.

## 6. Evidence appendix (what forced each decision)

All numbers processed-EM unless noted; full records under
`~/acec/logs/` and `~/acec/reports/`.

- Mechanism B dead: five verifier variants; calibrated logistic on belief
  features, question-disjoint val AUC **0.4883**, all-negative weights;
  best selection arm +1.2pp ns; `acec_zero_shot@8` *hurts* (−3.9pp).
- Mechanism C dead: stop_coverage_min swept 0.55→0.85 with **no behavioral
  effect** (mean retrievals 1.385–1.408 vs none 1.381–1.396); held-out 1024q
  Δ −0.001, 1003/1024 identical outcomes.
- Mechanism A alive: gold-entity injection on 512 audited bridge questions:
  frozen_r3 **+7.2pp** [+4.3,+10.2], acec_ep50 **+3.1pp** [+1.6,+4.9], at
  equal-or-lower cost, intervention rates 43–65%.
- Binding bottleneck: current belief binds the correct bridge entity on
  **20.3%** of questions — the a1 capture limit that door 1 exists to raise.
- Training curve: ACEC-shaped GRPO flat at zero across 100 episodes (slope
  −0.03 EM/100ep) while outcome-only rose then saturated under identical
  hyperparameters — reward signal, not optimizer, was the binding constraint.
- Sampling headroom real: oracle@8 − pass@1 = **+11.7 to +18.0pp**; 30–46
  questions per policy recoverable by selection alone.
- Answer-contract lesson: strict-format EM under-measured true EM ~25×;
  measurement now separates contract compliance from correctness — v8
  inherits that separation.
