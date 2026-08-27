# Paper Outline — BETA v0.1

> Status: narrative beta. The central thesis and section structure are ready for drafting; final multi-seed statistics, matched outcome-only seeds, and the training-matched group analysis remain pending.
>
> Scope: this document is a working paper blueprint, not submission-ready prose.

---

## 1. Working Title

### Primary

**Breaking Outcome Ties: Evidence-Coverage Rewards for Deep Multi-Hop Search**

### Alternatives

1. **Learning from All-Failure Rollouts in Multi-Hop Search RL**
2. **Rewarding Reachable Evidence: Coverage-Augmented RL for Deep Search Agents**
3. **When Evidence Coverage Helps Multi-Hop Search Agents**

---

## 2. Central Thesis

> Outcome-only group-relative RL cannot distinguish partial progress when all rollouts for a question fail. Completion-gated evidence coverage restores relative preferences in these answer-tied groups, producing consistent deep-hop gains without additional search. Its benefit is strongest when the required evidence is reachable and diminishes when open-corpus retrieval becomes the dominant bottleneck.

The paper is about **group-level answer-signal starvation and trajectory discrimination**, not about a new step-level advantage estimator.

### One-line Chinese summary

> 多跳搜索 RL 的关键困难不是所有轨迹都没有奖励，而是同题 rollout 经常同时答错；证据覆盖在终局答案奖励沉默时恢复了对部分进展的相对排序。

---

## 3. Claims and Non-Claims

### Claims supported by the current evidence

1. Completion-gated evidence coverage improves closed-pool multi-hop QA, with the largest and most consistent effect on 4-hop questions.
2. The deep-hop improvement does not come from issuing more searches; 4-hop search count is stable or lower.
3. Answer-only discrimination becomes increasingly scarce with hop depth, while coverage restores discrimination in many all-failure groups.
4. Open-corpus transfer is structure-dependent: the effect is strong on 2Wiki, small on HotpotQA, and approximately neutral on open-corpus MuSiQue.
5. Evidence-related signals can be useful as training rewards and offline diagnostics without being useful as runtime controllers or injected observations.

### Claims that must not appear

1. Non-potential coverage rewards uniquely “survive” a group-relative baseline.
2. The current estimator provides independent step-level or turn-level credit assignment.
3. Answer-tied groups have zero total gradient. They have zero **terminal-answer discrimination**, while cost and penalty terms may still differ.
4. Task gradients only enter action tokens. Free-form tokens still receive task gradients under the current configuration and are constrained by a stronger KL coefficient.
5. Coverage is a strong within-question predictor of immediate answer correctness. The corrected mixed-group analysis does not support this claim.
6. The current results establish universal open-world retrieval improvement.

---

## 4. Abstract Skeleton

1. **Problem.** Reinforcement learning for multi-hop search agents commonly relies on terminal answer rewards, but long-hop questions often yield groups of uniformly unsuccessful rollouts.
2. **Observation.** In such answer-tied groups, group-relative learning cannot use terminal correctness to distinguish trajectories that found partial supporting evidence from trajectories that made no meaningful progress.
3. **Method.** We augment trajectory returns with completion-gated first-hit evidence coverage, together with retrieval costs and channel-specific KL regularization for stable long-horizon optimization.
4. **Primary result.** Across `[FINAL_SEED_COUNT]` explicit seeds, coverage shaping improves `[FINAL_OVERALL_DELTA]` overall and `[FINAL_4HOP_DELTA]` on 4-hop questions, without increasing 4-hop search calls.
5. **Mechanism and boundary.** A training-matched group analysis shows that coverage expands discriminated all-failure groups most strongly at greater hop depth; transfer remains strong on structurally matched 2Wiki questions but weakens when open-corpus evidence retrieval is the bottleneck.
6. **Conclusion.** Evidence coverage is most useful as a training-time trajectory-ranking signal for deep search, rather than as a universal runtime controller or a replacement for retriever quality.

Do not finalize the abstract until:

- `[SEED3_PENDING]` is complete;
- `[PAIRED_NOSHAPE_SEEDS_PENDING]` is resolved;
- `[K8_V22_GROUP_ANALYSIS_PENDING]` replaces or validates the current K=4 v3 analysis.

---

## 5. Contribution List

### C1. Answer-signal starvation in group-relative search RL

Identify and quantify a failure mode in which terminal answer rewards cannot distinguish rollouts because an entire question group is correct or incorrect, with all-failure groups becoming dominant on long-hop questions.

### C2. Completion-gated evidence coverage as a trajectory-ranking signal

Introduce a deterministic, judge-free reward computed from first-time retrieval of gold supporting titles. The reward augments within-question trajectory ranking without claiming a new per-turn advantage estimator.

### C3. Seed-robust deep-hop improvement with no search inflation

Show that the benefit concentrates on 4-hop and retrieval-hard questions and is not explained by a larger search budget.

### C4. Applicability and signal-placement boundaries

Show that the effect depends on evidence reachability and compositional structure, and that evidence signals that work for training or diagnosis need not work as runtime control inputs.

C4 can be reduced to an analysis contribution if the main paper becomes too broad.

---

## 6. Paper Structure

## 1 Introduction

### Paragraph 1: why deep search RL matters

- Multi-hop questions require iterative evidence acquisition and integration.
- Agentic retrieval creates long trajectories with many possible partial successes and failures.
- Existing systems often optimize terminal answer correctness plus formatting and cost terms.

### Paragraph 2: the overlooked problem is group-level answer ties

- RLOO/GRPO learns from within-question rollout comparisons.
- On difficult questions, many or all rollouts receive the same answer outcome.
- The issue is not merely sparse reward over time; it is insufficient **within-group ranking resolution**.

### Paragraph 3: evidence annotations provide partial-progress discrimination

- Supporting-document titles are available during training.
- First-hit coverage provides a cheap and deterministic measure of evidence progress.
- Completion gating discourages trajectories that collect evidence but never attempt an answer.

### Paragraph 4: headline findings

- Multi-seed closed-pool improvement, concentrated on deep hops.
- Search calls remain stable or decrease.
- Group analysis shows the largest increase in discriminated all-failure groups at 4 hops.
- Transfer succeeds selectively, revealing an evidence-reachability boundary.

### Paragraph 5: contributions

Use C1–C4 above. Do not lead with the cross-system leaderboard.

---

## 2 Related Work

### 2.1 Reinforcement learning for search agents

- Outcome-based training for iterative retrieval and tool use.
- Position the unified baseline evaluation as context, not causal evidence for coverage shaping.

### 2.2 Process, path, and evidence rewards

- Step-wise query and retrieval rewards.
- Planner-aligned path coverage.
- Citation- and rubric-based evidence-chain rewards.
- Distinguish the present work by its exact, deterministic reward and its focus on group-level trajectory discrimination.

### 2.3 Group-relative learning and partial-progress supervision

- Explain that the paper studies when a group-relative estimator lacks terminal-answer discrimination.
- Do not claim that dense reward timing alone solves temporal credit assignment.

### Positioning sentence

> Unlike step-wise credit-assignment methods, we study how privileged evidence supervision restores within-group trajectory discrimination when terminal outcomes are tied.

---

## 3 Answer-Signal Starvation in Multi-Hop Search RL

### 3.1 Multi-hop search trajectories

Define a trajectory:

\[
\tau_i=(s_1,a_1,o_1,\ldots,s_T,a_T),
\]

where actions include search, answer, and termination-related protocol actions.

Clarify:

- closed-pool training environment;
- open-corpus evaluation environment;
- maximum turns and top-k retrieval;
- gold evidence is privileged training information, not a test-time requirement.

### 3.2 Group-relative trajectory learning

For K rollouts of the same question:

\[
A_i^{\mathrm{RLOO}}
=R_i-\frac{1}{K-1}\sum_{j\neq i}R_j.
\]

Define an **answer-tied group** as a group in which all rollouts have the same terminal correctness label.

When answer outcomes are tied, terminal correctness contributes no within-group ordering, although search cost, token cost, and other penalties may still differ.

### 3.3 Coverage restores partial-progress preferences

Let final evidence coverage be \(C_i\). Its contribution to the relative advantage is:

\[
A_i^{\mathrm{cov}}
=\beta\left(C_i-\frac{1}{K-1}\sum_{j\neq i}C_j\right)
=\beta\frac{K}{K-1}(C_i-\bar C).
\]

Interpretation:

- a common coverage offset cancels;
- group-level coverage differences remain;
- uniformly unsuccessful rollouts can therefore receive different preferences according to evidence progress.

This is a trajectory-ranking argument, not a claim of independent step-level advantages.

---

## 4 Coverage-Augmented Trajectory Ranking

### 4.1 Reward definition

\[
R(\tau)
=R_{\mathrm{answer}}
+\beta C_{\mathrm{evidence}}
-\lambda_s N_{\mathrm{search}}
-R_{\mathrm{format/timeout/token}}.
\]

Explain:

- alias exact-match answer reward;
- first-time gold-title hits;
- coverage normalization by the number of gold titles;
- completion/answer-attempt gate;
- retrieval, format, timeout, and free-token costs.

### 4.2 Incremental implementation and trajectory-level effect

- Coverage is emitted at first retrieval of each gold title.
- Under the current trajectory-level estimator, the principal effect is to alter trajectory returns and rankings.
- Do not describe the method as turn-level credit assignment unless a separate estimator is introduced and tested.

### 4.3 Channel-specific KL regularization

- Lower KL coefficient for protocol/action tokens.
- Stronger KL coefficient for free-form reasoning tokens.
- Present as a stability mechanism and implementation component.
- Do not call it independently necessary until a clean uniform-KL comparison is available.

### 4.4 Algorithm box

Suggested algorithm:

1. Sample K trajectories for each question.
2. Parse search and answer actions.
3. Track first-time gold evidence hits.
4. Apply completion gating and cost terms.
5. Compute trajectory returns and RLOO advantages.
6. Broadcast the trajectory advantage according to the implemented policy update.
7. Apply token-type-specific KL regularization.

---

## 5 Experimental Setup

### 5.1 Models and training

- Qwen3.5-9B base.
- SFT v2.2 initialization and full teacher provenance.
- LoRA rank, rollout count, batch size, learning rate, turn budget, and GPU budget.
- Distinguish available training-pool size from the number of unique questions actually visited.

### 5.2 Environments and datasets

- MuSiQue closed-pool training and evaluation.
- Wiki18 open-corpus MuSiQue.
- HotpotQA and 2WikiMultiHopQA transfer.
- Clearly separate closed-pool and open-corpus claims.

### 5.3 Controlled comparisons

Primary causal comparison:

- outcome-only RL;
- coverage-shaped RL;
- matched SFT initialization;
- matched seeds, training budget, data order, checkpoint rule, and evaluation protocol.

Secondary comparisons:

- zero-shot base;
- SFT-only;
- external search-RL systems under the unified retrieval environment.

### 5.4 Metrics and statistics

- EM and F1;
- 2/3/4-hop breakdown;
- search calls and invalid/give-up rates;
- seed-level mean and standard deviation;
- question-level paired wins/losses and McNemar or bootstrap statistics;
- explicit distinction between training variance and question-sampling uncertainty.

### 5.5 Checkpoint selection

- Specify a fixed checkpoint or a selection/reporting split.
- Do not select and report the best checkpoint on the same full development set without labeling the resulting selection bias.

---

## 6 Does Coverage Improve Deep Search?

### 6.1 Primary multi-seed result

Main table:

| Reward | Seeds | Overall EM | 2-hop | 3-hop | 4-hop | Searches | 4-hop Searches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Outcome only | `[N]` | `[MEAN±SD]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| + Coverage | `[N]` | `[MEAN±SD]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Current directional evidence relative to the existing outcome-only checkpoint:

| Shaped run | Overall ΔEM | 4-hop ΔEM | 4-hop ΔSearch |
|---|---:|---:|---:|
| Original run | +1.65 | +7.16 | -0.346 |
| Explicit seed 1 | +0.58 | +3.95 | -0.119 |
| Explicit seed 2 | +1.28 | +2.72 | -0.035 |
| Explicit seed 3 | `[PENDING]` | `[PENDING]` | `[PENDING]` |

Until matched outcome-only seeds are available, describe this as consistency across shaped reruns relative to a fixed outcome-only reference, not a paired multi-seed causal estimate.

### 6.2 Depth-dependent effect

- Plot seed-level ΔEM for 2/3/4-hop questions.
- Test whether the effect increases with hop depth.
- Emphasize the stable 4-hop direction even if the overall effect remains modest.

### 6.3 Accuracy versus search efficiency

- Report EM together with average search calls.
- Establish that 4-hop improvement is not purchased by issuing more searches.
- If possible, add accuracy-versus-search Pareto curves or matched search-budget analysis.

---

## 7 Why Does Coverage Help?

### 7.1 Current group-level evidence

The current analysis uses 240 stratified closed-pool questions, four samples per question, and a post-RL v3 policy. Treat it as preliminary until reproduced under the training-matched v2.2, K=8 setting.

Current corrected summary:

| Hop | Answer-discriminated groups | + all-wrong groups with coverage variation | Total | Multiplier |
|---|---:|---:|---:|---:|
| 2-hop | 25.0% | 2.5% | 27.5% | ×1.10 |
| 3-hop | 28.7% | 28.8% | 57.5% | ×2.00 |
| 4-hop | 15.8% | 30.8% | 46.7% | ×2.95 |

Interpretation:

> Coverage contributes little when answer rewards already distinguish rollouts; its principal role is to provide structured preferences among uniformly unsuccessful trajectories.

### 7.2 Statistics that should not be used

Do not use the original AUC values 0.8229/0.7325/0.7595. They were inflated by non-tie-aware rank handling.

Corrected tie-aware, cross-question within-hop AUC values are approximately:

- 2-hop: 0.606;
- 3-hop: 0.574;
- 4-hop: 0.621.

The more relevant within-question mixed-group AUC values are approximately:

- 2-hop: 0.559;
- 3-hop: 0.544;
- 4-hop: 0.484.

These do not support a claim that coverage reliably predicts immediate answer correctness within a question group.

### 7.3 Submission-grade mechanism analysis

Replace or validate the current result using:

- SFT v2.2 initialization;
- K=8 rollouts;
- temperature and environment matched to training;
- preferably SFT, intermediate, and final checkpoints;
- complete base reward, including costs and penalties;
- the fraction of groups for which coverage adds ranking information beyond the outcome-only return.

### 7.4 Desired conclusion

The mechanism section should establish:

1. long-hop groups are more often uniformly unsuccessful;
2. evidence progress still varies inside many of these groups;
3. coverage converts this variation into relative preferences;
4. the causal training ablation shows that these preferences improve the final deep-hop policy.

---

## 8 Where Does Coverage Transfer?

### 8.1 Open-corpus results

Current paired results:

| Setting | Shaped ΔEM | 4-hop ΔEM | ΔSearch |
|---|---:|---:|---:|
| Wiki18 MuSiQue ep100 | -0.70 | +0.25 | -0.281 |
| Wiki18 MuSiQue ep120 | +0.12 | +0.25 | -0.355 |
| Wiki18 HotpotQA ep100 | +0.10 | — | -0.191 |
| Wiki18 HotpotQA ep120 | +0.85 | — | -0.270 |
| Wiki18 2Wiki ep100 | +2.05 | +4.52 | -0.348 |
| Wiki18 2Wiki ep120 | +3.85 | +4.98 | -0.484 |

### 8.2 Interpretation

- Do not claim universal open-world improvement.
- Closed-pool and 2Wiki results support a deep/compositional benefit.
- Open MuSiQue suggests that coverage-shaped behavior cannot compensate for missing or unreachable evidence.
- HotpotQA is predominantly 2-hop in the present evaluation and shows only a small effect.

### 8.3 Reachability analysis

Desired decomposition:

\[
P(\text{correct})
=P(\text{evidence reached})
\times P(\text{correct}\mid\text{evidence reached}).
\]

Report:

- gold-title recall or evidence reachability;
- accuracy conditional on full and partial evidence coverage;
- failure due to retrieval versus failure after successful evidence acquisition;
- ΔEM as a function of hop depth and reachability.

The reachability explanation remains a hypothesis until this decomposition is measured.

---

## 9 Where Should Evidence Signals Enter the Agent?

### 9.1 Training reward

- Positive deep-hop causal effect.
- Main use of privileged evidence supervision.

### 9.2 Offline diagnostic

- Coverage and belief-related scores can support auditing, risk stratification, and selective prediction.
- Keep diagnostic metrics separate from the within-group training mechanism.

### 9.3 Runtime controller or injected observation

- Runtime reranking/control does not produce a reliable gain.
- Observation injection underperforms its matched control and increases inference cost.
- The lesson is not that evidence information is useless, but that predictive information does not automatically become a causally useful control input.

### Framing sentence

> Evidence supervision can be a good training signal and a useful diagnostic without being a good runtime controller.

If space is limited, keep only the summary figure in the main paper and move J7/J8 implementation details to the appendix.

---

## 10 Limitations

State explicitly:

1. gold supporting titles are privileged training annotations;
2. SFT v2.2 uses a large teacher and its provenance must be disclosed;
3. the main causal training environment is a closed candidate pool;
4. current open-corpus benefits are dataset- and structure-dependent;
5. exact-title matching may not transfer to tasks without document-level supervision;
6. not every available MuSiQue training question is visited during the current RL schedule;
7. channel-specific KL is not yet isolated by a clean final-quality ablation;
8. checkpoint selection and seed-level uncertainty must be reported separately.

---

## 11 Conclusion Skeleton

1. Terminal answer rewards are frequently unable to distinguish rollouts for deep multi-hop questions.
2. Completion-gated evidence coverage supplies relative preferences in many all-failure groups.
3. This produces stable deep-hop improvements without search inflation.
4. The benefit depends on evidence reachability and should not be interpreted as universal open-world retrieval improvement.
5. Privileged evidence is most useful as a training-time trajectory-ranking signal rather than a general runtime controller.

---

## 12 Figure Plan

### Figure 1 — Central mechanism figure

Two panels:

- **Panel A:** K rollouts with identical answer outcome but different retrieved evidence; outcome-only cannot rank by correctness, while coverage supplies partial-progress preferences.
- **Panel B:** answer-discriminated versus answer-or-all-wrong-coverage-discriminated group fractions across 2/3/4 hops.

Caption claim:

> Evidence coverage restores group-level trajectory discrimination primarily on long-hop questions where terminal answer outcomes are uniformly incorrect.

### Figure 2 — Multi-seed deep-hop effect

Two panels:

- seed-level ΔEM across 2/3/4 hops;
- corresponding ΔSearch.

Caption claim:

> Coverage shaping produces its most consistent gains on 4-hop questions without increasing the number of searches.

### Figure 3 — Transfer and reachability boundary

Preferred final form:

- x-axis: evidence reachability or gold-title recall;
- y-axis: shaped-minus-outcome-only ΔEM;
- markers: closed MuSiQue, open MuSiQue, HotpotQA, and 2Wiki;
- color or shape: hop depth.

Temporary form before reachability is computed:

- grouped ΔEM and ΔSearch bars for the four evaluation settings.

### Figure 4 — Signal placement map, optional

Three placements:

- training reward: positive;
- offline diagnostic: useful;
- runtime controller/injected observation: no improvement or negative.

Move to the appendix if it competes with the central mechanism story.

---

## 13 Table Plan

### Table 1 — Primary matched multi-seed result

Outcome-only versus coverage-shaped RL under identical seeds and training conditions.

### Table 2 — Clean same-base ladder

Under one retrieval environment:

1. base zero-shot;
2. SFT;
3. outcome-only RL;
4. coverage-shaped RL.

Do not mix closed-pool and open-corpus results in one causal ladder.

### Table 3 — Open-corpus and transfer boundary

MuSiQue, HotpotQA, and 2Wiki with EM/F1, hop breakdown, search count, and evidence reachability.

### Table 4 — Unified external-system comparison

Secondary result or appendix. Make model-size, training-data, and retrieval-environment caveats explicit.

---

## 14 Writing Order

### Start immediately

1. Create the paper project and section skeleton.
2. Write Figure 1 and Figure 2 captions before drawing the figures.
3. Draft Sections 3–5: setup, estimator analysis, method, and experimental protocol.
4. Draft the main-results section around placeholders for seed 3 and matched outcome-only seeds.

### After seed 3 finishes

1. update the primary result table;
2. regenerate the hop-level effect figure;
3. decide whether the headline should emphasize overall EM or 4-hop EM;
4. freeze checkpoint and seed terminology.

### After matched outcome-only seeds and K=8 analysis

1. finalize causal mean±SD;
2. replace the preliminary K=4 v3 group analysis;
3. freeze the abstract numbers;
4. write Introduction and Related Work;
5. write the abstract last.

---

## 15 Pre-Submission Evidence Checklist

### Required

- [ ] Explicit seed 3 completes and is evaluated.
- [ ] Matched outcome-only seeds are completed or the causal scope is explicitly narrowed.
- [ ] Group analysis is repeated with SFT v2.2 and K=8.
- [ ] AUC implementation uses correct tie handling; unsupported AUC claims are removed.
- [ ] “Zero gradient” is replaced with “zero terminal-answer discrimination.”
- [ ] Main causal ladder uses one base, SFT branch, retriever, corpus, prompt, and budget.
- [ ] Teacher provenance and actual unique training-question exposure are disclosed.
- [ ] Checkpoint selection is separated from final reporting or honestly labeled.
- [ ] Open-corpus no-shape comparisons are included.
- [ ] All central figures are generated from versioned scripts and traceable prediction files.

### Strongly recommended

- [ ] Evidence-reachability/error decomposition is complete.
- [ ] Incremental versus terminal coverage is compared or the trajectory-level interpretation is stated explicitly.
- [ ] Uniform versus channel-specific KL is isolated if KL is retained as a major contribution.
- [ ] Training compute, rollout tokens, search calls, and wall time are reported.
- [ ] J7/J8 details are moved to the appendix unless signal placement remains a central contribution.

---

## 16 Narrative Decision Rule

> **RESOLVED (2026-08-28, author decision): draft now under Branch 2 — 4-hop is
> the primary endpoint of the causal claim.** Overall EM is still reported
> everywhere (tables, ladder, per-seed paired deltas with 3/3 direction
> consistency), but the abstract/intro stake the shaping claim on deep-hop
> gains without search inflation. Reversible: if the paired no-shaping seeds
> (in queue, ~09-01) yield significant overall-EM paired deltas across pairs,
> promote overall EM back per Branch 1. Writing is unblocked on this basis —
> do not wait for pending results to start Sections 1/3–5.

Use the following final framing after all pending results arrive:

- If paired multi-seed gains remain positive overall and on 4-hop: lead with **Breaking Outcome Ties**.
- If only 4-hop remains robust: lead with **Deep-Hop Targeted Improvement**, make 4-hop the primary endpoint, and avoid broad average-performance claims.
- If evidence reachability strongly explains transfer: use **Rewarding Reachable Evidence** as the title and central boundary result.
- If runtime signal-placement results remain the most distinctive finding: retain them as a major analysis contribution, but do not let them displace the core training mechanism without additional matched experiments.

