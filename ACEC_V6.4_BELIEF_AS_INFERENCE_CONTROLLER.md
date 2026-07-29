# ACEC v6.4 — Belief as an Inference-Time Controller (the EM-winning program)

Status: **implemented locally, not yet validated on the two-H100 runtime.**
Additive companion to
`ACEC_V6.2_EVOLUTION_DIRECTION.md` (belief/reward evolutions) and
`ACEC_V6.3_EVIDENCE_OUTPUT_AND_JOINT_EVALUATION.md` (evidence output + official
Joint metric, implemented). V6.3's harness measures whether the belief *selects*
better evidence post-hoc; v6.4 asks the harder question v6.3 cannot: can the
belief, *driving actions in the inference loop*, make ACEC **more accurate than
frozen R3 and matched outcome-only** on held-out answer EM/F1 — not just more
efficient.

The current outcome-only run, v1–v6.3 files, and the v6.3 evaluation harness
remain unchanged. V6.4 is implemented only in new inference-loop/controller
entry points until it passes the gates in Section 10. “EM-winning” is the
research objective, not a result claimed before those gates.

---

## 1. The reframe: the belief's EM value is as a controller, not a reward shaper or an evidence citer

Three uses of the belief have been on the table:

- **reward shaper** (v5/v6.2 E1/E2) — shapes the *retrieval* sub-policy during
  GRPO; established to buy retrieval efficiency, not answer EM, because the dense
  signal never reaches the answer tokens (v6.2 Section 2);
- **evidence citer** (v6.3) — selects/emits `S` for the official SP/Joint metric;
  measures and audits, but does not by itself change what the policy answers;
- **inference-time controller** (v6.4) — the belief *drives* what to retrieve
  next, when to stop, and which answer to trust.

The design doc's C2 ("coverage as a policy input is empirically dead, ≤ ±0.3pt")
was measured with **weak interventions on the frozen policy** (belief vector,
prefix injection, thresholds, budgets). That is not the strong form. The strong
form is not "feed the belief as a feature"; it is "let the belief **select the
action**": gap-directed query generation, adaptive retrieval budget, and
verifier-based answer selection. These are active control policies, not features.

**Sequencing consequence (the core method of v6.4):** every mechanism below is
first tested **at inference time on the existing ep50 checkpoint, with zero
policy training.** If a belief-driven controller has no oracle headroom, no
binding signal, and no held-out inference lift, long GRPO is not justified yet.
This is not the absolute claim that training can never discover a signal absent
from the frozen controller; it is a compute-allocation gate. If a controller
does lift EM, that is positive evidence for baking the same action path into
training.

---

## 2. Existing signals versus genuinely new machinery

The belief already computes the following controller inputs:

- `belief.suggest_target_slot()` (`coverage_belief.py:182`) — the lowest-`p`
  unbound slot = the **retrieval gap** (Mechanism A).
- `bind_slot` / `slot.bound_entities` / `augment_hypothesis_with_bound_entities`
  — **bridge-entity binding** for query injection (Mechanism A).
- `belief.should_stop_voi()` (`coverage_belief.py:191`) — the **stop signal**
  (Mechanism C).
- `belief.coverage()` / `coverage_variance()` — **coverage/confidence features**
  that may help answer verification (Mechanism B); they are not already a
  calibrated trajectory-correctness verifier.
- `TurnResult.suggested_target_slot` / `.stop_voi` already surface these per turn.

V6.4 nevertheless adds three pieces of new machinery and evaluates each
separately:

1. a **belief-in-the-loop inference path** that changes the next query or stop
   action;
2. fixed-K question decomposition and slot pre-seeding, which changes the
   belief runtime and therefore requires a no-controller/K=1 harm check;
3. a new answer-correctness verifier artifact, fitted only on
   question-disjoint answer labels. V6.3's evidence NLI is the generic baseline,
   not a pre-existing ACEC correctness verifier.

---

## 3. Cheap prerequisites and ceilings

1. **B0: measure oracle@N before fitting any verifier.** For every policy report
   `pass@1/2/4/8` (oracle EM/F1) and unique-answer count. If sampling never
   produces additional correct answers, verifier work cannot recover them.
2. **A0: re-derive the bridge oracle** (replaces the unverified +27.9pt figure; see
   the 2026-07-22 provenance caveat in `ACEC_Belief_RAG_design.md`). On the
   current pipeline (R3-RAG-Qwen + wiki18 + processed EM), for bridge-typed
   held-out questions, inject the *gold* bridge entity (from `supporting_facts`)
   into the hop-2 query and measure processed-EM lift vs no injection. This
   sizes Mechanism A's ceiling before any effort is spent chasing it. Run the
   K=1 oracle comparison at temperature 0 so the first-hop policy is
   deterministic; B0 separately uses nonzero temperature to measure sampling
   headroom.
3. **E5-only check.** Fix the E5 slot artifact
   (`ACEC_V6.2_EVOLUTION_DIRECTION.md` E5). Binding
   fires only when a slot's `p` crosses `bound_threshold`, which requires the
   slot to survive several turns. With empty-start slots and DECOMPOSE spawning a
   fresh slot every turn (~1.44 retrieval turns/trajectory), most slots get one
   observation and never bind — the code-level reason bridge rewriting "fired 0
   times" historically. Pre-seeding K slots from a question decomposition lets
   slots accumulate coverage and bind. This is a belief-runtime change, not
   policy training. Because this is new machinery, first compare
   `preseed=off` versus `preseed=on + monitor-only` at K=1: binding precision
   must be meaningful and answer EM must not regress before it is allowed to
   steer retrieval.

---

## 4. Mechanism A — gap-directed retrieval (retrieval-steering lever)

**Idea.** The belief knows the unresolved requirement (`suggest_target_slot`) and
any bound bridge entities. Use them to steer the next query: generate a query
explicitly targeting the lowest-`p` unbound slot, grounded by bound entities
(bridge-entity injection is the special case). This changes *what information
enters the context* on hop 2+, which is the largest structural lever for a
fixed-reasoning policy — it lets ACEC retrieve evidence R3's implicit query never
finds.

**Inference-time form (no training).** Between turns, if the belief has a bound
entity for the target slot, inject it into the policy's next retrieval query
(prompt-level append or slot-targeted rewrite). This is the design's D8–10
"posterior-driven bridge-entity rewriting with zero training," now unblocked by
Section 3.2.

**Ablations.**
- gold-entity oracle injection (upper bound, Section 3.1) —
- belief-learned injection (the deployable result) —
- R3 implicit (no injection, baseline).

Run A0 (gold vs none) before A1 (learned vs none). A1 is interpretable only
after E5 reports binding rate, oracle-entity binding precision/recall, and the
fraction of trajectories on which query text actually changed.

**Measurement.** hop-2+ retrieval recall lift; processed-EM and official
Answer/Joint EM **on the bridge-typed slice specifically** (comparison-typed
questions carry little bridge headroom and dilute the signal). The claim is: the
belief-learned injection recovers a meaningful fraction of the re-derived oracle.

---

## 5. Mechanism B — belief-as-verifier best-of-N (test-time scaling lever)

**Idea.** Sample `K` trajectories/answers from the frozen policy; score each with
the belief's grounding + coverage (`coverage()`, and the v6.3 answer-conditioned
grounding of the emitted answer by selected evidence); select the best. Verifier-
guided best-of-N is the most reliable way to beat a fixed policy on EM, and
ACEC's belief is a purpose-built verifier.

**Inference-time form (no training).** First report oracle@N. Then compare a
generic answer-conditioned NLI score with (a) a fixed zero-shot ACEC score and
(b) an immutable ACEC answer-correctness calibrator fitted on a
question-disjoint split. The calibrator uses grounding, coverage, and
coverage-confidence; it never consumes gold supporting-fact membership. No
policy update is involved.

**Ablation ladder (isolates the belief's value).**
- `K=1` — R3 baseline;
- `K=N` + self-consistency (majority vote) — no belief;
- `K=N` + generic answer-conditioned NLI verifier — the v6.3 NLI arm as verifier;
- `K=N` + ACEC belief verifier — grounding + coverage.

A result of `belief-verifier > NLI-verifier > majority-vote`, with material
oracle@N headroom and a verifier calibration gate that passes, is a clean,
publishable contribution **independent of the reward-shaping thesis**.

**Measurement.** processed/official EM lift vs `K`; report the **EM × compute**
frontier using both trajectory count and actual model-generation calls
(best-of-N and controller-forced final answers can change them); that trade is
the point here and is reported as a Pareto rather than hidden.

---

## 6. Mechanism C — coverage-gated adaptive budget (allocation lever)

**Idea.** Instead of uniformly fewer retrievals (the v5 efficiency framing),
allocate the budget by answerability: stop early when `should_stop_voi()` fires
on high coverage; keep retrieving when coverage/answerability is low. Net budget
is neutral or lower, but reallocated toward the hard questions where R3 gives up
— turning an efficiency mechanism into an accuracy mechanism.

`should_stop_voi()` is currently a **coverage VOI heuristic**, not an estimator
of expected answer-EM gain. It therefore has lower priority than A/B and must
not be described as answer-value calibration.

**Inference-time form (no training).** Replace the policy's implicit stop with
the belief-gated stop; calibrate thresholds on a disjoint validation sweep and
freeze an artifact whose mean retrieval cost matches the declared reference
within tolerance. Measure held-out cost again rather than assuming the match
transfers.

**Ablations.** fixed-budget R3 stop / uniform-early ACEC stop / coverage-gated
adaptive stop, all at matched mean retrieval cost.

**Measurement.** EM on the low-base-rate hard slice; verify total retrieval cost
is neutral-or-lower so the gain is allocation, not spend. Directly resolves the
"efficiency vs EM conflict" flagged in v6.2.

---

## 7. The unification: belief as a VOI / POMDP controller

A+C combined, principled. The belief is a calibrated per-slot, per-action
posterior over coverage. At each step, estimate the expected answer-value (EM
gain) of each available action — retrieve-for-slot-`j`, rewrite-with-entity-`E`,
answer-now — and take the argmax (a value-of-information policy over the belief
state). This is the decision-theoretic version of gap-direction + adaptive
budget in one controller, and it is what the "action-conditioned" name promises:
a principled belief-state controller competing with an RL-trained black-box
policy. Pursue this only after A and C show inference-time lift separately; it is
the ambitious unification, not the first experiment.

---

## 8. Sequencing — cheapest falsification first

1. **B0 oracle@N** on frozen R3, outcome-only, and ACEC ep50.
2. **A0 bridge oracle** on an assessable bridge-only slice.
3. **E5-only monitor:** preseed off/on, controller disabled, K=1.
4. **B1 verifier ladder:** first / majority / NLI / ACEC zero-shot /
   calibrated ACEC.
5. **A1 learned injection:** only if E5 binding passes.
6. **C adaptive stop:** only with a disjoint cost-matching artifact.
7. Combine only independently positive levers; then consider GRPO.

The clean experiment is a policy × controller × verifier × K factorial:

- policy: frozen R3 / matched outcome-only / ACEC ep50;
- controller: none / monitor-preseed / belief gap / adaptive / combined
  (gold entity is an oracle, never a deployable arm);
- verifier: K=1 first / majority / generic NLI / ACEC;
- K: 1 / 2 / 4 / 8.

This prevents a better base policy, a new slot decomposition, and a selector
from being collapsed into one “ACEC” number.

`run_scripts/41_infer_controller_ep50_v64.sh` encodes this order as explicit
`RUN_PROFILE` values: `b0` (default), `a0`, `e5_off`, `e5_on`, `b1`, `a1`, `c`,
and `full`. The expensive `full` profile is intentionally not the default.
`b1/full` require a passing verifier artifact; `c/full` require a disjoint
cost-matched controller artifact; `full` also requires matched outcome-only;
`a0` requires the audited bridge manifest.
Environment variables can still override controller/K/temperature choices, and
every effective value is written to `results.json`. Profiles that request only
`first` selection skip the batch answer-NLI pass entirely; verifier profiles
record the exact number of NLI pairs as part of test-time compute.

This is the same discipline as v6.3 (measure on checkpoints we already own before
spending GPU hours) applied to the answer axis.

---

## 9. Evaluation discipline (what makes the EM claim publishable)

Under the v6.3 official protocols (official `hotpot_evaluate_v1.py` for Answer/
SP/Joint; processed EM/F1 as the R3-comparable diagnostic; short-answer contract
so official EM is not depressed by verbosity):

- **Statistical power / seeds.** The n=256 single-seed CI (half-width ≈ 4.3 pt on
  processed EM; worse on Joint, an intersection event) cannot resolve the ≥2-EM
  target. Every headline EM claim requires **multiple seeds and/or a larger
  held-out set**, with paired bootstrap CIs on fixed question ids.
- **OOD.** Report Bamboogle (~125 Q), 2Wiki, and MuSiQue held-out — the regime
  where the belief's calibration-only (not per-example-SF) supervision should
  generalize where a title-matching reward cannot. Bring one OOD adapter into the
  pipeline (only HotpotQA exists today); the v6.3 evidence contract's dataset
  adapters are the seam for this.
- **Isolating ablations.** Each mechanism ships with the ablation ladder that
  attributes the gain to the belief and not the scaffold: A = oracle / learned /
  none; B = K=1 / majority / NLI-verifier / belief-verifier; C = fixed / uniform /
  adaptive at matched cost. A mechanism without its ladder is not reportable.
- **Pareto, not a point.** Report the **EM × retrieval-cost × test-time-compute**
  frontier across arms, not a single EM number: A and C move retrieval cost, B
  moves compute. The claim is a dominating frontier, which is robust to the
  single-number metric fragility the project has repeatedly hit.
- **Matched baselines.** The paper target is not merely `ACEC > frozen R3`.
  ACEC must beat both frozen R3 and the strongest matched outcome-only policy
  under the same controller/verifier/compute budget. The inference result
  writes paired EM/F1 bootstrap deltas against both references.
- **Official and processed axes.** The short-answer contract makes official
  Hotpot Answer EM primary. R3-style processed EM/F1 remains a separately
  labelled diagnostic and is run from saved predictions on CPU/API; it is not
  silently substituted for official EM.

---

## 10. Gates and isolated implementation

New inference-loop/controller files; v1–v6.3 files and the v6.3 eval harness are
not edited during validation.

### Gate 0 — Sampling and bridge ceilings
Report oracle@N/pass@N before verifier fitting. Re-derive the bridge oracle on
the current pipeline (bridge slice, gold-entity injection, Answer and processed
EM with/without), using only automatically assessable, audited entity labels.

### Gate 1 — E5 and inference contracts
The monitor-only E5 preseed must improve binding coverage, report entity
precision/recall where an oracle exists, preserve K=1 EM within uncertainty,
and preserve a true no-op baseline. The inference path must enforce strict
short-answer parsing, isolate gold annotations from runtime, and log every
controller intervention.

### Gate 2 — Mechanism B (best-of-N), ep50, no training
Full oracle/majority/NLI/ACEC ladder and EM × compute Pareto. A calibrated ACEC
verifier is usable only if its question-disjoint AUC and top-1 oracle-recovery
gate passes.

### Gate 3 — Mechanism A (gap-directed retrieval), ep50, no training
Oracle vs learned vs none; hop-2 recall + bridge-slice EM with paired CIs.

### Gate 4 — Mechanism C (adaptive budget), ep50, no training
Question-disjoint threshold sweep, immutable passing cost-match artifact,
held-out fixed/uniform/adaptive stop ablation, hard-slice EM and actual cost.

### Gate 5 — Bake the winning lever(s) into training
Only levers that showed inference-time lift; 1-episode contract check, then
25/50 episodes (healthy horizon), with the v6.2 over-optimization controls.

### Gate 6 — Multi-seed + OOD + Pareto
The publishable result: seeds, Bamboogle/2Wiki/MuSiQue, and the frontier of
Section 9, under the official metric protocols.

### Planned new files
```text
rag/src/belief/acec/inference_controller_v64.py     # gap-direction, adaptive stop, VOI
rag/src/belief/acec/answer_verifier_v64.py          # belief-as-verifier for best-of-N
rag/train/infer_belief_controller_v64.py            # belief-in-the-loop inference path
rag/train/fit_answer_verifier_v64.py                 # answer-only, disjoint calibration
rag/train/fit_controller_artifact_v64.py             # disjoint cost-match threshold freeze
run_scripts/40_bridge_oracle_rederive_v64.py        # Gate 0 oracle re-derivation
run_scripts/41_infer_controller_ep50_v64.sh         # Gates 2-4, existing checkpoint
run_scripts/42_eval_v64_processed_em_api.sh          # cached R3 processed diagnostic
rag/train/tests/test_inference_controller_v64.py
rag/train/tests/test_answer_verifier_v64.py
```
Every controller artifact records schema/version, git commit, consumed belief
outputs, verifier/injection mode, seeds, and the frozen thresholds — same
discipline as the v6.3 selector artifacts.

The local implementation is complete at the contract/static-test level. It is
not evidence that ACEC already beats R3: that conclusion starts only after the
two-H100 B0/A0/E5 runs and the matched policy factorial above.
