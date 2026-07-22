# ACEC v6.2 — Evolution Direction

Status: design proposal, not yet implemented. Written 2026-07-22 after reading
the full v3→v4→v5 experimental arc, the v5 marginal-utility standard, the v5
runtime (`rag/src/belief/acec/runtime_v5.py`), and the 2026-07-22 processed-EM
and episode diagnostics. This document deliberately sets aside the "next step"
sections written in the individual experiment notes and reasons from the data.

The numbering (v6.2) marks this as the belief/reward evolution proposal; the
decisive-experiment plan in Section 5 is the companion piece that must run
regardless of whether any belief change lands.

Implementation note: the isolated `grpo_rsf_vllm_v6.py` scaffold adds the
Section-6 observability and over-optimization controls (trajectory traces,
wrong/low-coverage answer guard, answer-turn KL anchoring, adaptive KL).  It
does **not** implement E1/E2 below and must not be presented as the v6.2 belief
evolution.  Its 100-episode run is intentionally sequenced after the clean
v5-vs-outcome-only comparison, so those controls cannot confound the decisive
ablation.

---

## 1. Where we actually are after v5

Three findings, in descending order of certainty:

1. **Answer accuracy is at statistical parity with the frozen model.** Under the
   proper R3 processed-EM protocol (Qwen2.5-72B extractor + normalized EM), the
   best checkpoint (episode 50) scores processed EM 0.3945 vs frozen R3 0.3867
   — **+0.78 pt, 95% CI [−3.5, +5.1]**. Processed F1 is slightly worse (0.5008
   vs 0.5192). The earlier raw-EM jump (0.094 → 0.367) was almost entirely a
   formatting effect: the extractor rescued 75 answers for frozen R3 but only 7
   for episode 50.

2. **The one surviving positive signal is retrieval efficiency.** Episode 50
   holds/improves Gold-SF recall (0.73 vs 0.70) with **−18.3% retrieval calls**
   (1.46 vs 1.79). Consistent across versions, single-subset, not yet multi-seed.

3. **There is over-optimization after episode 50.** Episodes 75/100: online
   reward keeps rising while held-out processed F1 *significantly* drops, with
   `who`/`what` questions losing 9–16 F1 points and individual answers flipping
   correct→wrong→correct across checkpoints. This is proxy/generalization
   mismatch — a soft reward-hacking signature.

Mechanism red flag: **78.7% of all retrieval actions are DECOMPOSE**
(`runtime_v5.py:205` spawns a new slot on every DECOMPOSE). The
"action-conditioned" observation model — the core novelty vs a pooled model —
is barely exercised for EXPAND (6.3%) and REWRITE (15%).

Two structural gaps confirmed absent from the entire repo (grep over
`experiments/`, `run_scripts/`, `rag/train/`):

- **No outcome-only / sparse-reward GRPO baseline has ever been run.** The
  thesis is "dense coverage reward densifies *sparse outcome* reward," yet every
  comparison to date is ACEC vs gold-SF (itself a dense process reward) or vs
  the frozen model. The baseline the claim depends on does not exist.
- **ACEC has only ever been trained/evaluated in-domain on HotpotQA**, where
  gold-SF holds a privileged supporting-fact label channel for its reward. The
  regime where ACEC structurally differs (label-free / OOD) has never been run.

---

## 2. Root cause — why ACEC ties on answers

Read the v5 reward directly (`runtime_v5.py:290`):

```
reward = R_ans (answer turn only) + eta·(coverage_after − coverage_before) + efficiency
```

The coverage term `eta·ΔC` fires **only on retrieval turns**. On the answer turn
`coverage_before == coverage_after` (`runtime_v5.py:192`), so ΔC = 0 and the
answer tokens receive **only the sparse R_ans**.

**Consequence: the dense belief signal shapes the retrieval sub-policy and never
touches the answer sub-policy. On the answer axis, ACEC is identical to
outcome-only by construction.** "Better evidence in context" and "correct answer
string from that evidence" are different skills; ACEC supervises only the first.
This is not a bug — it is the direct explanation for the cross-version pattern
"better retrieval evidence, same final answers."

Two amplifiers:

- **Coverage ≠ answerability.** High NLI support to a slot hypothesis does not
  mean the answer-critical entity/relation is surfaced. v4 already found a
  supporting document can have zero marginal utility. The proxy saturates before
  answerability does.
- **Potential-shaping is deliberately weak.** With γ=1 the coverage reward
  telescopes to `eta·(C_T − C_0)`, upper-bounded per trajectory by eta = 0.3
  against R_ans = 1.0. Policy-invariance requires it not dominate — which also
  caps how far it can move the answer axis.

---

## 3. Belief/reward evolutions that target the root cause

Ordered by leverage against the Section 2 diagnosis. E1 and E2 are the only
changes that can plausibly move the **answer** axis; E3 is a curriculum move; E4
is the honest fallback.

### E1 — Coverage measures answerability gain, not topical support (highest leverage)

Switch the utility provider from "requirement coverage" (current HotpotQA v5,
NLI/span support) to **"answer score"** — already a sanctioned provider in the v5
standard (Section 2.2): `U = frozen model's normalized EM / F1 / gold-answer
log-probability`, evaluated before and after adding the document. Every retrieval
reward then aligns to "did this document raise the probability of the correct
answer," not "is this document topically on-target."

- **Offline (calibration):** compute `ΔU_t` with the gold answer available — a
  frozen model's gold-answer log-prob delta when document `d_t` is added.
- **Runtime (no gold):** the runtime feature must be a *gold-free* answerability
  estimate. Use the v5 standard's frozen-judge provider (Section 2.3): a
  versioned evaluator estimates answerability before/after the document. This is
  the harder half and the main implementation risk — the runtime proxy must
  track the offline answer-score target well enough to calibrate against it.
- **Why this is the right first move:** it attacks the exact "proxy saturates
  before answerability" failure, and the standard already anticipates it, so it
  is a genuine belief evolution, not a calibration-v6 tweak.

### E2 — Put dense signal on the answer turn itself (groundedness reward)

Today the answer head only ever sees sparse R_ans (identical to outcome-only).
Add a reward on the **answer turn** tied to whether the emitted answer is
entailed/supported by the belief's covered evidence set — the first dense
supervision the answer generation ever receives. Complementary to E1: E1 makes
retrieval reach answer-critical evidence, E2 rewards the answer head for actually
using it. Guard against a faithfulness/format hack (rewarding fluent grounded-
sounding text that is still wrong) by keeping R_ans dominant and the groundedness
term as a shaping bonus.

### E3 — Curriculum on the low-base-rate slice (data, not belief)

At ~40% base answer rate, a warm-started policy plus G=8 rollouts already yields a
rich group-relative advantage from *sparse* reward alone — so on easy questions
outcome-only learns fine and there is no gap for dense shaping to win. Dense
coverage has room only where the sparse signal is genuinely sparse: the
low-base-rate deep multi-hop slice (the `what`/`who` types that degrade most).
Up-sample training/evaluation toward that slice; stop competing on the
high-base-rate questions where a tie is structurally guaranteed.

### E4 — Bank the efficiency win (fallback, already earned)

If E1/E2 do not open an answer-axis win, reframe the contribution honestly as
"equal accuracy, materially fewer retrievals, via a calibrated coverage-based
stopping signal that sparse or gold-label rewards cannot provide," and evolve the
VOI stop / coverage-guarded stopping to widen the −18% retrieval margin while
holding recall. This is the highest-probability real contribution and does not
require beating anyone on EM.

---

## 4. Which battlefield to win on

"Evolve ACEC to win" has two different answers depending on the terrain:

- **Win in-domain HotpotQA answer EM vs outcome-only** — via E1/E2. This is the
  **hard road**: it fights the structural fact that at high base rate sparse
  reward already suffices, and the potential-shaping ceiling. Honest prior:
  *probably a tie on answers*, an efficiency win on retrieval. Even a real 2-pt
  effect is unresolvable at n=256 single-seed (CI half-width ≈ 4.3 pt).

- **Win the thesis** — move to the regime gold-SF cannot occupy: **label-free
  training and OOD** (Bamboogle / 2Wiki / MuSiQue). There gold-SF's reward
  channel does not exist (no SF labels) while ACEC's calibrated belief still
  runs. This is the **aligned road**: it matches ACEC's only structural reason to
  exist (label-free training), and stacks with the efficiency signal that has
  survived every version.

Recommendation: pursue E1 to attack the answer axis, **but do not stake viability
on the in-domain answer-EM win.** Put the decisive comparison on the label-free /
OOD + efficiency battlefield, which is where ACEC is structurally advantaged.

---

## 5. The decisive experiments (must run regardless of belief changes)

These map directly onto the original design doc's own — never-executed —
decision thresholds.

### 5.1 Complete the in-domain baseline matrix
Add an **outcome-only GRPO arm**: R_ans only, coverage weight 0, no belief. Same
harness, 100 episodes, eval every 25 under processed EM. Yields the four-way
comparison under one protocol: frozen / outcome-only / gold-SF / ACEC. One
training run (~3.7 h) + one eval. This is the single most information-dense run
available and it has never been done.

### 5.2 Label-withheld training (the actual thesis test)
A HotpotQA split with supporting-facts hidden, forcing gold-SF to degrade to
outcome-only (no process reward) while ACEC runs its calibrated belief. If ACEC
beats degraded-gold-SF here, this is the paper's core claim.

### 5.3 OOD held-out eval (cheap down payment)
Evaluate the **existing** episode-50 adapters (ACEC vs gold-SF vs frozen) on
Bamboogle (~125 questions), 2Wiki, MuSiQue. No training — tests whether the
ACEC-trained adapter generalizes differently OOD. Bring one OOD dataset into the
pipeline (only HotpotQA exists today).

### 5.4 Statistical power
Processed EM at n=256 single-seed has CI half-width ≈ 4.3 pt — it cannot resolve
the design doc's ≥2-EM target. Any answer-axis claim needs multi-seed and/or a
larger held-out set. Characterize the extractor's determinism first: identical
inputs disagreed on processed EM in two duplicate groups during the 2026-07-22
run, so the headline metric has an uncharacterized noise floor.

---

## 6. Controls and sequencing (do not confound the decisive runs)

From the episode diagnostic, but re-sequenced:

- **Per-question trajectory logging — do first, as an enabler.** Lightweight
  JSONL of query / docs / NLI scores / stop position per rollout. Prerequisite
  for diagnosing the retrieval→answer conversion gap and auditing the DECOMPOSE
  monoculture, and for interpreting OOD results. Keep it a few log lines; do not
  rebuild a DuckDB/notebook analytics platform for it.
- **Extractor caching + determinism audit.** Cache by `(question, prediction)` to
  survive the multi-arm × multi-checkpoint × multi-seed eval sweep; but the real
  action item is explaining the temp-0 nondeterminism, because processed EM is
  the headline metric.
- **Over-optimization controls — AFTER the clean comparison, not before.**
  Coverage-guarded stopping is cheap/low-risk (inference-time, no gradient
  change) and may be included. Answer-token masking, adaptive KL, and SFT
  rehearsal change training dynamics and would confound the outcome-only /
  gold-SF / ACEC attribution — run the clean comparison at the healthy ~50-episode
  horizon first, then add these only to extend the horizon.

On the recurring warm-start / expert-trajectory idea: the v5 online curve
(0.16 → 0.54) is direct proof there is **no cold-start pathology** to fix. A
500-example SFT phase framed as "teach it answers up front" solves a non-problem.
If expert trajectories are used at all, their correct role is a small rehearsal
buffer against the episode-75/100 answer drift — i.e. a Section-6 over-opt
control, sequenced *after* the decisive comparison, not a training precursor.

---

## 7. Explicit non-goals

- **Do not build calibration v6.** The belief already passes its gates (AUC
  ~0.88, stable training). What is untested is the thesis, not the calibrator.
  E1 is a change of *what the belief measures*, not a further refinement of *how
  well it measures the same thing*.
- **Do not scale to 500/800 episodes on the unchanged recipe.** Episode 50 is the
  healthy checkpoint; 75/100 already over-optimize.
- **Do not rest a ≥2-EM claim on n=256 single-seed.**
- **Do not report raw/direct EM as a semantic result.** Use processed EM/F1 under
  one protocol, plus the independent semantic ACC judge, and name the protocol
  precisely against any external number.

---

## 8. Success criteria (mapped to the original design thresholds)

The paper is viable if any of:

- **Accuracy leg:** ACEC > outcome-only by ≥ 2 processed EM in-domain, at
  adequate power (multi-seed). Honest prior: unlikely — expect a tie (Section 4).
- **Efficiency leg:** ACEC = outcome-only / gold-SF on processed EM while using
  materially fewer retrievals, multi-seed. Highest-probability real contribution.
- **Label-free / OOD leg:** ACEC > degraded-gold-SF under label-withheld training
  (5.2), or ACEC-trained adapters generalize better than gold-trained OOD (5.3).
  Structurally the strongest claim, because it is the one thing gold-SF cannot do.

Honest fallback if ACEC ties everywhere including label-free: the design doc's
own Findings-tier framing — "process rewards for RAG are potential-based shaping;
here is the theory and the calibrated-belief generalization." Still publishable,
and only reachable *after* the label-free regime is actually tested — which is
the whole point of running Section 5.
