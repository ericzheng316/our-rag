# ACEC V7 — Belief-Conditioned Learnable Set Selection

Status: implemented locally as an isolated V7 path. V1–V6.4 files remain
unchanged.

## 1. Research contract

V7 tests three separable hypotheses:

1. With the same query, candidate pool, and context budget, a state- and
   set-conditioned selector chooses evidence with greater downstream answer
   value than retriever relevance top-K or a fixed relevance/coverage mixture.
2. Conditional coverage gain has independent value. If removing
   `conditional_delta_coverage` does not hurt, the result is a generic learned
   reranker rather than evidence for ACEC's coverage mechanism.
3. After freezing a shared selector, ACEC query GRPO produces better candidate
   pools than matched outcome-only/R3 policies. The selected-set potential
   change is assigned to the query that produced the pool; predicted gain is
   never paid as a second reward.

The main method is **answer-supervised and supporting-fact-label-free**.
Supporting-fact labels are permitted only in evaluator/oracle sidecars.

## 2. Scope

Kept from earlier versions:

- V5 calibrated belief artifact and K strategy;
- V6 turn-level GRPO, correct rollout temperature/log-probability semantics,
  KL anchoring, and answer guard;
- V6.3 official Answer/SP/Joint evaluation;
- V6.4 controller as a later factorial arm.

New in V7:

- wide retrieval pool, default `L=20`;
- fixed context selection budget, default `K=5`;
- pure conditional belief simulator;
- signed answer-value artifact;
- sequential Plackett–Luce set selector;
- answer-utility counterfactual slate data;
- frozen-selector query GRPO.

Not in V7.0:

- semantic hard filters;
- learned K/STOP;
- joint online updates of query policy, selector, and belief/value heads;
- new corpus/index;
- use of SF membership in selector training.

## 3. Runtime transition

For query action `q_t`:

```text
q_t
  -> retrieve D_t^L
  -> score all candidates against frozen belief
  -> sequentially select S_t^K
  -> update live belief once with S_t^K
  -> put S_t^K in model context
  -> assign selected-set potential change to q_t
```

Unselected documents do not update the live belief and do not enter the
language-model context. Candidate pools also exclude documents selected in
earlier turns, and offline replay obeys the same cumulative context capacity
as the online rollout. If fewer than K positions remain, that state's
effective K is recorded as `selection_capacity` and reused when constructing
counterfactual slates.

The current action labeler mutates its query history. V7 therefore performs
routing, scoring, selection, and belief update in one transaction. A
probe-then-update implementation is forbidden because it can silently change a
REWRITE action into EXPAND.

## 4. Conditional set value

The selector state is:

`x_t = (b_t, q_t, turn, remaining_budget)`.

Candidate `d` is scored conditional on the already selected prefix `S`:

`delta_C_hat(d | S) = C(T(b_t, S union {d})) - C(T(b_t, S))`.

The simulator reproduces the live turn semantics: for each slot it chooses the
document with maximum raw entailment, then applies that document's frozen
action-conditioned hit posterior. It never mutates the live belief, and runtime
fails closed if simulated and live post-selection coverage differ by more than
`1e-6`.

Because current coverage is monotone, it cannot represent corrective negative
evidence by itself. V7 therefore exposes:

- conditional coverage gain;
- belief entropy/information change;
- signed downstream answer value;
- entailment, contradiction, and neutral probabilities;
- retriever score/rank;
- slot alignment and selected-set interaction;
- state coverage/uncertainty/binding/budget.

Contradiction and overlap are features, not fixed penalties. Every valid,
unique candidate retains non-zero probability during stochastic selection.

## 5. Selector

Default architecture:

- two-layer MLP, hidden size 128;
- state/set-conditioned feature vector;
- sequential Plackett–Luce selection without replacement;
- train temperature 0.7;
- deterministic greedy top-K at evaluation;
- less than one million parameters.

Fixed relevance top-K and fixed `relevance + coverage` are baselines only.

## 6. Answer-value supervision

For gold answer `a*`, the external frozen answerer supplies:

`u(S) = mean_log_p(a* | question, history, S)`.

Every sampled slate prefix is scored, giving a signed marginal label:

`delta_V*(d | S) = u(S union {d}) - u(S)`.

Question ids are split deterministically into 70% fit, 15% validation, and 15%
internal test. Fit and validation question hashes are stored in every artifact.
Feature/metadata contracts reject SF membership fields.

Default slate data per state:

1. relevance top-K;
2. fixed relevance/coverage greedy;
3. stochastic slate at temperature 1.0;
4. stochastic slate at temperature 1.5.

Their selection propensities are logged for later IPS/SNIPS/DR analysis.

## 7. Selector loss

V7 supports candidate-level listwise training and set-level counterfactual
policy-gradient training. The default slate objective uses the within-state
mean utility as its baseline:

`L = -E[(u(S)-mean_state_u) log rho(S|x)]`

with:

- KL to retriever prior: 0.02;
- selector entropy coefficient: 0.01;
- pairwise slate preference coefficient: 0.5.

The answer-value and belief heads are frozen/stop-gradient while the selector
is fitted.

## 8. Query reward and PBRS modes

There is one environment reward:

`R = R_answer + eta(Phi(s') - Phi(s)) - retrieval_cost`.

`delta_C_hat` participates in selection but is never added as another bonus.
`grpo_estimator_v7.py` rejects a non-zero `predicted_gain_bonus`.

Two explicit modes are logged:

- `legacy_coverage_aux`: V6-compatible coverage auxiliary objective;
- `strict_pbrs`: adds `-eta * terminal_coverage` when entering the absorbing
  terminal state.

The selector is frozen throughout V7.0 query GRPO. Joint hierarchical
query/selector advantages are deferred to V7.1.

## 9. Gates

### G0 — contracts and leakage

- no SF label in selector/value training payloads;
- deterministic split and artifact hashes;
- simulator/live coverage error <= `1e-6`;
- selector leaves live belief unchanged.

### G1 — candidate headroom

- `recall@20 - recall@5 >= 5` absolute points;
- answer-utility/oracle headroom is positive and large enough to plausibly
  recover at least 1.5–2 Answer EM points.

If G1 fails, stop selector work: the bottleneck is query/index/corpus.

### G2 — signed value

- held-out pairwise accuracy >= 0.58;
- rank correlation >= 0.20;
- answer-correctness AUC >= 0.65 when the binary audit is available;
- selected-tail ECE <= 0.12.

### G3 — offline selector

- held-out set win rate over relevance >= 55%;
- paired bootstrap lower bound > 50%;
- positive mean signed answer-utility gain;
- `full - no_delta_C >= 2` win-rate points for an ACEC coverage claim.

### G4 — frozen inference

- 256-question pilot does not regress Answer EM by more than one point;
- confirm on at least 1,024 questions/multiple seeds;
- Answer EM/F1 improves by at least 1.5 points with paired lower bound > 0;
- matched candidate/context/retrieval budgets;
- p95 selector overhead <= 50%.

`infer_frozen_v7.py` is the answer/retrieval Gate-4 runner. It records selected
document ids and titles but does not fabricate Hotpot sentence ids for
`wiki18_100w` chunks. Official distractor SP/Joint remains the native
provenance V6.3 evaluation path; it is a separate report, not silently mixed
into the V7 answer gate.

### G5 — one-episode training smoke

- 16 questions x 8 samples;
- no OOM/NaN;
- no simulator/live mismatch;
- 100% selection traces;
- format error < 1%;
- validated engine/HF log-probability alignment.

### G6 — query GRPO

- run 25, then 50, then 100 episodes;
- selector/value/belief artifacts remain frozen;
- stop at 50 if held-out direction is not positive;
- do not run 500/800 episodes until the 100-episode frozen-selector comparison
  is positive.

### G7 — paper claim

At least three seeds and 1,024 fixed held-out questions. V7 must beat matched
outcome-only and R3 under the same selector/controller/verifier/compute by at
least two Answer EM/F1 points with paired bootstrap lower bound above zero.
The no-coverage-feature ablation must be weaker for the ACEC-specific claim.

## 10. Implementation map

Core:

- `contracts_v7.py`
- `belief_simulator_v7.py`
- `answer_value_v7.py`
- `set_selector_v7.py`
- `selector_loss_v7.py`
- `runtime_v7.py`

Offline pipeline:

- `collect_candidate_pools_v7.py`
- `assign_question_splits_v7.py`
- `build_counterfactual_sets_v7.py`
- `score_counterfactual_answer_value_v7.py`
- `fit_answer_value_v7.py`
- `train_set_selector_v7.py`
- `eval_candidate_headroom_v7.py`
- `eval_set_selector_v7.py`
- `summarize_selector_eval_v7.py`
- `eval_selected_calibration_v7.py`
- `compare_selector_delta_c_ablation_v7.py`
- `validate_artifacts_v7.py`
- `validate_smoke_v7.py`
- `infer_belief_selector_v7.py`
- `infer_frozen_v7.py`

Training:

- `grpo_estimator_v7.py`
- `grpo_rsf_vllm_v7.py`

Entry scripts:

- `50_build_v7_counterfactual.sh`
- `51_score_v7_answer_value.sh`
- `52_fit_v7_answer_value.sh`
- `53_train_v7_selector.sh`
- `53b_train_v7_no_delta_c_ablation.sh`
- `54_eval_v7_selector.sh`
- `54c_compare_v7_delta_c_ablation.sh`
- `54b_infer_v7_frozen_checkpoint.sh`
- `55_validate_v7_one_episode_two_h100.sh`
- `56_train_v7_100epi_two_h100.sh`

The selected-tail calibration audit uses the maximum frozen
action-conditioned slot-hit posterior as its probability and title-level
support membership as its evaluator-only Bernoulli label. Conditional
coverage gain is deliberately not treated as a calibrated probability.

## 11. Required per-turn log

- complete candidate ids/base rank/base score;
- slot entailment/contradiction/neutral probabilities;
- conditional coverage and information gain;
- signed answer value;
- every sequential probability/log-probability and chosen order;
- selected and unselected ids;
- belief before/after and simulator error;
- the single current-turn reward;
- artifact/split/git hashes;
- retrieval/NLI/selector/generation latency.

This log is the audit trail for winner's curse, exposure bias, self-confirmation,
and query-versus-selector credit attribution.

## 12. Execution order

1. `50_build_v7_counterfactual.sh` — frozen replay, split, L/K slates, recall
   headroom.
2. `51_score_v7_answer_value.sh` — frozen answer likelihood and complete G1.
3. `52_fit_v7_answer_value.sh` — runtime artifact plus five-fold OOF bundle.
4. `53_train_v7_selector.sh` — full selector.
5. `53b_train_v7_no_delta_c_ablation.sh` — matched no-delta-C selector.
6. Run `54_eval_v7_selector.sh` once for each selector on the `test` split.
7. `54c_compare_v7_delta_c_ablation.sh` — paired ACEC-mechanism gate.
8. Only if the held-out selector, selected-tail calibration, and delta-C
   ablation reports all pass, run `54b_infer_v7_frozen_checkpoint.sh`.
9. Run `55_validate_v7_one_episode_two_h100.sh`; then 25/50 checkpoints before
   `56_train_v7_100epi_two_h100.sh`.

The GPU launchers require the three independent G2/G3 reports and save their
combined preflight result as `v7_preflight_gates.json`.
