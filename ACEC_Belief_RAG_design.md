# ACEC-Belief RAG: Action-Conditioned Evidence-Coverage Belief for Agentic Multi-Hop RAG

**Design document, v1.0 — 2026-07-02.**
Target: core technical contribution for ACL/EMNLP/NeurIPS/SIGIR-class submission, built on the existing R3-RAG + BAMDP codebase (HotpotQA real-retrieval baseline: EM 43.2 / Judge 62.8, avg_docs 15.87).

---

## 0. Critique of the premise first: what I rejected, what I kept

Your prompt asks me to push back before formalizing. Four pushbacks, each grounded in your own logs:

**(C1) Action-indexing the *existing* system-level Betas is the weak version of Gap A — reject it.** Your diagnostics show `θ_ret` saturates at E[θ]≈0.65–0.75 with thresholds (0.85/0.92) firing 0–1 times in 7405 samples. The root cause is not action-agnosticism; it is that a *global competence parameter pooled across a stationary retriever and corpus* concentrates at the corpus base rate and carries near-zero per-question information. Splitting its pseudo-counts by action (`w_EXPAND·o` vs `w_REWRITE·o`) re-indexes a feature that is already uninformative. Reviewers would (correctly) call this "action as a bookkeeping tag." The strong version changes **the object of belief**: from *component quality* to *per-question, per-slot evidence coverage*. Action-conditioning then becomes structural — it determines *which slot's* filter an observation routes to and *which likelihood* applies — and the pooled update is recoverable as a degenerate special case (Section 1.4). That is a theorem-shaped statement, not a feature flag.

**(C2) Coverage as a *policy input* alone is empirically dead — the reward channel must be primary.** Every inference-time belief intervention you ran on the frozen policy was ≤ ±0.3pt (belief vector, prefix injection, thresholds, budgets), because R3-RAG-Qwen already stops implicitly (mean 1.62 turns; 8.5% answer at step 0). The measurable headroom is elsewhere: retrieval recall 49.5%, and +27.9pt EM when the bridge entity is injected into the hop-2 query. So ACEC's center of gravity is **credit assignment during GRPO training** — a dense, calibrated, per-turn coverage signal that teaches the policy *what to retrieve next* (slot-targeted, bridge-entity-explicit queries) — with stopping/efficiency as the secondary win. Belief features to the policy stay in the design but as an ablated channel, not the headline.

> **[2026-07-22 provenance caveat — do not treat +27.9pt as established.]** The
> +27.9pt bridge-injection figure quoted above has **no reproducible source in
> this repository** — no script, experiment record, or data produces it (grep of
> `experiments/`, `run_scripts/`, and all code returns only design-doc
> references). It appears to be a pre-ACEC oracle diagnostic (inject the *gold*
> bridge entity into the hop-2 query, measure EM lift = an upper bound, not an
> achieved result), likely measured on an earlier belief system / model /
> retriever / eval subset, and may **not transfer** to the current
> R3-RAG-Qwen + wiki18 + processed-EM pipeline. Any strategy that prioritizes
> the bridge-entity lever on the strength of this number must first **re-derive
> it on the current pipeline** (bridge-typed held-out questions, gold bridge
> entity from `supporting_facts`, injected into hop-2, processed-EM with vs
> without). This is the cheap probe the design's own D8–10 already specified and
> that was never run — the mechanism fired 0 times historically (see
> `CLAUDE.md`), plausibly because the empty-start / DECOMPOSE slot artifact
> suppresses entity binding (see `ACEC_V6.2_EVOLUTION_DIRECTION.md` E5). Treat
> the magnitude as unknown until re-measured; the qualitative point (retrieval
> steering is the largest structural lever for a fixed-reasoning policy) stands
> independently of the number.

**(C3) A single Beta over "coverage fraction" is the wrong distribution family — reject it.** Coverage fraction is not the parameter of an exchangeable Bernoulli process: slots are few, non-exchangeable, monotonically absorbed (once covered, always covered), and their *number* is unknown. The right family is a **structured posterior**: independent per-slot binary latents with monotone Bayesian filtering, plus a categorical posterior over the number of required slots $K$. A Dirichlet-multinomial is also wrong (slots are not draws from a shared urn).

**(C4) Demote $\theta_{llm}, \theta_{noise}$; reframe $\theta_{diff}$.** Keep the Beta-Bernoulli machinery only where it earns its keep: action-indexed *hit-rate* parameters $\pi_a$ (Section 1.3) and, optionally, the legacy slots as ablation. $\theta_{diff}$ becomes the amortized prior over $K$ — the one place "difficulty" has a crisp operational meaning (how many evidence pieces are required).

What survives from your working definition: the BAMDP frame, the two-layer (doc→slot) belief with NLI bridging (it becomes the observation model), the train/test asymmetry treatment via `supporting_facts` as distant supervision, and the demand that the whole thing be GRPO-compatible both forward and backward.

---

## 1. Problem formalization

### 1.1 Hyperstate, actions, observations

An episode is a multi-turn rollout of the policy LLM $\pi_\phi$ (R3-RAG-Qwen) on question $q$. At turn $t$ the model emits analysis text and either a search query or a final answer; a query triggers retrieval of $D_t$ (top-10 passages).

**Latent task structure.** Each question has an unknown requirement set $Z = \{z_1,\dots,z_{K^*}\}$: the minimal evidence pieces needed to answer (for HotpotQA, the `supporting_facts`; $K^*{=}2$ typically; MuSiQue up to 4). Neither $Z$ nor $K^*$ is observed at test time.

**Working slots.** The agent maintains identified slots $S_t = \{\hat s_1,\dots,\hat s_{K_t}\}$: sub-goal descriptions seeded by the query decomposer (split server) and grown when a discovered bridge entity spawns a new dependency. Each $\hat s_j$ carries a declarative hypothesis template $\mathrm{hyp}_j$ (e.g., *"[Shirley Temple] held government position X"*), which may be **bound** (dependencies resolved) or **unbound**.

**Coverage latents.** For each slot, $c_{j,t} \in \{0,1\}$: *the evidence pool $E_t$ contains a passage resolving $\hat s_j$*. Coverage is monotone: $c_{j,t+1} \ge c_{j,t}$ (docs are never dropped).

**Belief.** $b_t = \big(\kappa_t,\ \{p_{j,t}\}_{j\le K_t},\ \{(\alpha_a,\beta_a)\}_{a\in\mathcal{A}}\big)$ where $\kappa_t(k) = P(K^*{=}k \mid q, \text{history})$ on $\{1..K_{\max}\}$, $p_{j,t} = P(c_{j,t}{=}1 \mid o_{1:t})$, and $(\alpha_a,\beta_a)$ are Beta parameters over action-indexed hit rates. The **hyperstate** is $\tilde s_t = (s_t, b_t)$ with $s_t = (q, H_t, E_t)$ as in your current design. Because the filter (Section 1.3) is a deterministic function $\tau(b_t, a_t, o_t)$, the process over hyperstates is Markov — the BAMDP is an MDP on $\tilde s$, which is what licenses the shaping result in Section 3.

**Action space.** $\mathcal{A} = \{(\mathsf{m}, j) : \mathsf{m} \in \{\text{EXPAND}, \text{REWRITE}, \text{DECOMPOSE}\},\ j \le K_t{+}1\} \cup \{\text{ANSWER}\}$. The LLM emits free text; a deterministic **labeler** $\lambda(\text{query}_t) \to a_t$ assigns the abstract action: target $j = \arg\max_j \cos(\mathrm{E5}(\text{query}), \mathrm{E5}(\hat s_j))$ if the max $\ge \tau_{new}$, else DECOMPOSE (spawn $\hat s_{K_t+1}$); mode EXPAND if $j$ was previously targeted with a paraphrase-similar query, REWRITE if the query contains a novel entity binding for $j$. The policy remains token-level; $a_t$ is a measurable function of its output — no constrained decoding is required, which keeps ACEC drop-in compatible with R3-RAG's free-form format.

**Observation.** $o_t = (D_t, M_t)$ with $M_t[j] = m_{j,t} = \max_{d \in D_t} \mathrm{NLI}(d \Rightarrow \mathrm{hyp}_j) \in [0,1]$: the strongest entailment any *new* doc gives slot $j$'s hypothesis, computed by the cross-encoder already present in your two-layer architecture (the per-doc scores are your doc-level beliefs; the max over docs is the noisy-OR bridge to slot level).

### 1.2 Generative model of observations (action-indexed)

Per turn, each slot either receives a **hit** ($h_{j,t}{=}1$: a new doc resolves it) or not, with prior hit probability depending on the action:

$$h_{j,t} \sim \mathrm{Bernoulli}\big(\pi_{a_t}^{\rho(j,a_t)}\big), \qquad \rho(j,a_t) = \begin{cases} \mathsf{tgt} & j = \text{target}(a_t)\\ \mathsf{inc} & \text{otherwise (incidental)}\end{cases}$$

and the NLI observation is emitted from class-conditional densities that also depend on role and binding status:

$$m_{j,t} \sim f^{\rho,\, \mathrm{bound}(j)}_{h_{j,t}}(\cdot)$$

$f_1, f_0$ are 1-D densities (histogram or two-parameter logistic) **fitted offline on labeled trajectories** (Section 2.2). This is the formal content of "action-conditioned": the action indexes both the hit prior $\pi_a$ and the likelihood family through the routing role $\rho$. Different action modes are different *assays* of the same latent $c_j$ — EXPAND re-probing a stale slot has a lower hit prior than a REWRITE that injects a fresh bridge entity — which is what makes failures attributable (query-side vs corpus-side) instead of pooled into one confounded global parameter. **Pooled updating — your current implementation — is exactly the special case $f^{\mathsf{tgt}} = f^{\mathsf{inc}}$, $\pi_a \equiv \pi$, $K_t \equiv 1$**, which collapses the filter to a single global quality track and reproduces the saturation you observed.

### 1.3 Belief update $\tau(b_t, a_t, o_t)$ — closed form

Per-turn hit posterior (Bayes on one Bernoulli):

$$g_{a}(m_{j,t}) \;=\; \frac{\hat\pi_a\, f_1(m_{j,t})}{\hat\pi_a f_1(m_{j,t}) + (1-\hat\pi_a) f_0(m_{j,t})}, \qquad \hat\pi_a = \frac{\alpha_a}{\alpha_a+\beta_a}$$

Monotone coverage recursion (posterior of an OR of per-turn hits, conditionally independent given actions):

$$\boxed{\;p_{j,t} \;=\; 1 - (1 - p_{j,t-1})\,\big(1 - g_{a_t}(m_{j,t})\big)\;}$$

$K$ posterior: initialized $\kappa_0 = p_\psi(K\mid q)$ (amortized predictor); a DECOMPOSE that spawns slot $K_t{+}1$ truncates and renormalizes $\kappa_{t+1}(k) \propto \kappa_t(k)\,\mathbb{1}[k \ge K_t{+}1]$.

Action-indexed hit-rate Betas (the surviving, now action-indexed, Beta-Bernoulli machinery — optional within-episode adaptation, default ON with capped effective sample size to avoid the posterior-feeding-itself circularity): $\alpha_a {+}{=}\, g_a(m),\ \beta_a {+}{=}\, 1-g_a(m)$ for the targeted slot only; priors $(\alpha_a^0, \beta_a^0)$ fitted offline per mode.

**Coverage functional and uncertainty** (unidentified slots $j > K_t$ have $p_{j,t} := 0$ — requirements you haven't found are uncovered by definition, which is what makes early bridge discovery *increase* $C$ later):

$$C_t \;=\; \mathbb{E}[\text{fraction covered}] \;=\; \sum_{k} \kappa_t(k)\, \frac{1}{k} \sum_{j \le k} p_{j,t}, \qquad \mathrm{Var}[C_t] = \underbrace{\sum_k \kappa_t(k) \frac{1}{k^2}\sum_{j\le k} p_{j,t}(1-p_{j,t})}_{\text{within-}K} + \underbrace{\mathrm{Var}_{\kappa_t}\!\Big[\tfrac{1}{K}\sum_{j\le K}p_{j,t}\Big]}_{\text{across-}K}$$

### 1.4 Why this is "more than adding the action as a feature" — the three formal roles

1. **Routing:** $a_t$ determines which slot receives the $\mathsf{tgt}$ likelihood; the transition kernel $\tau(\cdot, a, \cdot)$ itself differs across actions, satisfying the BAMDP's action-indexed observation model rather than annotating a shared one.
2. **Identifiability:** with pooled likelihoods, {bad query, missing corpus fact, already-covered slot} are observationally confounded; action-indexed links (different $\pi_a$, different $f^\rho$) disentangle them the way multiple assays identify one parameter.
3. **Credit:** the shaped reward $\eta\,\Delta C_t$ (Section 3) carries the action label of the turn that produced it — per-action credit assignment is only meaningful because $\tau$ is action-indexed.

---

## 2. Core algorithm

### 2.1 One rollout turn (inference-time; no gold labels anywhere)

```
ALGORITHM 1: ACEC turn (executed after every policy turn t)
Input: belief b = (κ, {p_j}, {(α_a, β_a)}), slots S with hypotheses & bindings,
       policy output y_t (analysis + query | answer), new docs D_t
Params (all fitted offline, frozen at test): NLI cross-encoder, f1/f0 densities,
       hit priors (α⁰_a, β⁰_a), K-predictor p_ψ, thresholds τ_new, τ_para

1  if y_t is ANSWER: return b, C(b), STOP
2  a_t ← λ(y_t)                        # labeler: (mode, target j) via E5 cosine vs slots
3  if a_t is DECOMPOSE: S ← S ∪ {new slot}; κ(k) ∝ κ(k)·1[k ≥ |S|]   # renormalize
4  bind hypotheses: substitute entities confirmed in covered slots into hyp_j
5  for each slot j:                     # 10 docs × |S| slots NLI, one batch, ~20 ms
6      m_j ← max_{d∈D_t} NLI(d ⇒ hyp_j)
7      ρ ← tgt if j = target(a_t) else inc
8      g ← π̂_a·f1^{ρ,bound}(m_j) / (π̂_a·f1^{ρ,bound}(m_j) + (1−π̂_a)·f0^{ρ,bound}(m_j))
9      p_j ← 1 − (1 − p_j)(1 − g)
10     if ρ = tgt: α_a += g; β_a += 1−g          # capped ESS
11 C ← Σ_k κ(k)·(1/k)·Σ_{j≤k} p_j
12 features ← [C, √Var(C), E[K], H(κ), p_1..p_Kmax (padded), bound flags, t]
13 return b′, ΔC = C′ − C, features       # ΔC → reward; features → policy (optional channel)
```

Newly discovered bridge entities (line 4) also parameterize the *suggested next query* for the lowest-$p_j$ unbound slot — this is the mechanism aimed directly at your +27.9pt bridge-injection headroom, now driven by the posterior instead of an oracle.

**Interpretable stopping diagnostic (not the trained policy).** The myopic value-of-information rule
$$\text{STOP if } \eta \cdot \max_{(\mathsf{m},j)} \; \omega_j \,(1-p_{j,t})\, \hat\pi_{(\mathsf{m},j)} \;<\; c_r, \qquad \omega_j = \mathbb{E}_{\kappa_t}\!\big[\tfrac{1}{K}\mathbb{1}[j\le K]\big]$$
is the reward-consistent one-step-optimal stop and doubles as an inference-time gate on the frozen policy (Week-2 experiment). It is exact as a myopic expectation — under the generative model of Section 1.2, $\mathbb{E}_m[g_a(m)] = \hat\pi_a$ (the prior-predictive mean of a posterior equals the prior), so $\mathbb{E}[\Delta p_j \mid a] = (1-p_{j,t})\hat\pi_a$ with no approximation. It is myopic — it undervalues DECOMPOSE, whose payoff arrives via later slots — so it is a diagnostic lower bound, not a replacement for the learned policy.

### 2.2 Offline fitting (the train side of the train/test asymmetry)

From existing logged trajectories (your 7405-sample `records.jsonl` replays suffice — no new GPU inference):

- **Gold hit labels:** turn $t$, slot $j$ scores a hit iff new docs contain a gold `supporting_facts` passage whose title Hungarian-matches slot $j$ (E5 similarity between slot description and SF title/sentence). This is exactly your Phase-2 `r_sf_marginal` machinery, reused as supervision.
- **Fit:** $f_1^{\rho,\mathrm{bound}}, f_0^{\rho,\mathrm{bound}}$ as score histograms (or Platt fits) split by role and binding; $\hat\pi_a$ as empirical hit rates per action mode; $p_\psi(K|q)$ as a 2-layer MLP on the frozen E5 embedding of $q$ (labels: $|SF|$ counts; MuSiQue gives 2–4 spread).
- **Auxiliary losses** (all outside the policy gradient): $\mathcal{L}_{hit}$ = BCE of $g$ against gold hits; $\mathcal{L}_K$ = CE of $p_\psi$; calibration enforced by construction (class-conditional densities) and audited by ECE.

At test time the computation is *identical* with frozen parameters — the only asymmetry is that the observation model's parameters were estimated on labeled trajectories. There is no feature the training-time filter sees that the test-time filter does not. This is the strongest available answer to the reviewer question you flagged: the design does not "use SF at training and improvise at test"; it uses SF to *calibrate a fixed observation model*, then runs one filter everywhere.

---

## 3. GRPO integration — reward, advantages, and the credit-assignment story

### 3.1 Reward decomposition as potential-based shaping

Define the potential over hyperstates $\Phi(\tilde s_t) := C_t \in [0,1]$ and the per-turn reward

$$r_{t} \;=\; \underbrace{\mathbb{1}[t{=}T]\,R_{ans}}_{\text{sparse outcome}} \;+\; \underbrace{\eta\,\big(\Phi(\tilde s_{t+1}) - \Phi(\tilde s_t)\big)}_{R_{cov}\text{: dense coverage shaping}} \;-\; \underbrace{c_r\,\mathbb{1}[a_t \ne \text{ANSWER}]}_{R_{eff}} \;(+\; R_{fmt})$$

with defaults $R_{ans} \in \{0,1\}$ (EM, or 0.5·EM+0.5·Judge), $\eta = 0.3$, $c_r = 0.05$. The legacy $R_{ret} = \Delta\theta_{ret}$ term is **dropped, superseded**: $\Delta\theta_{ret}$ rewards "retrieval got better at anything" and empirically saturates; $\Delta C$ rewards "retrieval got the thing this question still needs."

Because the BAMDP is an MDP over hyperstates and $\Phi$ is a function of the hyperstate only, $R_{cov}$ is **potential-based shaping in the sense of Ng, Harada & Russell (1999): it leaves the optimal policy invariant for any $\Phi$, including a miscalibrated one** — miscalibration costs variance and learning speed, never the optimum. This is the formal wedge against the 2025–26 sufficiency/redundancy reward bonuses (Section 5), which are additive heuristics that *do* move the optimum and therefore must be hand-tuned against reward hacking. With $\gamma{=}1$ the shaping telescopes: $\sum_t R_{cov,t} = \eta(C_T - C_0)$.

**A unification worth stating in the paper:** the gold process reward you planned for Phase 2, $r^{sf}_t = |\text{new SF titles at } t|/|SF|$, is *exactly* $\Delta\Phi^*$ for the gold-coverage potential $\Phi^* = $ (fraction of SF covered). Prior process-reward RAG systems have been doing potential-based shaping without noticing; ACEC makes the potential an explicit, calibrated posterior — which is what buys the invariance guarantee, the uncertainty estimates, and a test-time object.

### 3.2 Why trajectory-level GRPO would silently destroy the dense signal — and the fix

This subtlety is load-bearing; it belongs in the paper. Standard GRPO computes one advantage per rollout from total return, $A_i = (R_i - \bar R)/\sigma_R$, applied to every token. Under $\gamma{=}1$ telescoping, shaped total return is $R_i + \eta(C_{T_i} - C_0)$. Within a GRPO group all rollouts share the question, so $C_0$ is identical and **cancels in the group baseline**; only $C_{T_i}$ survives. Hence under trajectory-level advantages, all of ACEC's dense shaping collapses to a *final-coverage bonus* — the per-turn structure is invisible to the gradient. Dense rewards require **turn-level advantages**:

$$G_{i,t} = \sum_{t' \ge t} r_{i,t'}, \qquad A_{i,t} = \frac{G_{i,t} - \mu_t}{\sigma_t + \varepsilon}, \qquad \mu_t, \sigma_t \text{ over rollouts with } T_i \ge t$$

applied to all tokens of turn $t$ in rollout $i$ (fallback to trajectory-level stats when fewer than 2 rollouts remain alive at $t$; turn-level grouping follows the GTPO line of multi-turn credit-assignment work). The GRPO objective is otherwise standard (clipped ratio, group size $G{=}8$, KL to reference):

$$\mathcal{J}(\phi) = \mathbb{E}\Big[\tfrac{1}{G}\sum_i \tfrac{1}{|y_i|}\sum_{t,\,\text{tok}\in t} \min\big(\rho_{tok} A_{i,t},\, \mathrm{clip}(\rho_{tok}, 1{\pm}\epsilon) A_{i,t}\big)\Big] - \beta\,\mathrm{KL}(\pi_\phi \| \pi_{ref})$$

### 3.3 Hybrid reward: answering "why not just use the gold process reward?"

At training time on SF-labeled data, gold $\Delta\Phi^*$ is available — lower variance and unhackable, so **use it where it exists**. The belief-side $\Delta C$ earns its place three ways: (i) it is the *test-time object* — policy feature, slot-targeting signal, and stopping gate exist only because the posterior is computable without gold; (ii) it enables **RL on SF-free data** (NQ, synthetic multi-hop, Bamboogle-style distributions) with the *same* reward semantics, which is the scaling story; (iii) it credits *semantic* coverage — alternative passages that entail a required fact but don't title-match SF (a known false-negative mode of title-based process rewards). Default: $r_{cov} = \Delta\Phi^*$ on labeled prompts, $\eta\Delta C$ on unlabeled prompts, with $\mathcal{L}_{hit}$ keeping the two aligned in distribution; ablate gold-only / belief-only / hybrid.

### 3.4 What is learned where (the differentiability decision, stated as a decision)

| Component | Parameters | Training signal | Differentiable through? |
|---|---|---|---|
| Policy $\pi_\phi$ (7B, LoRA) | $\phi$ | GRPO token-level policy gradient | yes — the only gradient path |
| NLI observation head, $f_1/f_0$, $\hat\pi_a$, $p_\psi(K\|q)$ | $\psi$ | supervised aux losses on logged trajectories ($\mathcal{L}_{hit}, \mathcal{L}_K$), **fitted before RL, frozen during RL** | no (by choice) |
| Belief update $\tau$ | none | closed-form probability arithmetic | no (by construction) |

The belief module is deliberately **part of the environment/harness from GRPO's perspective**: it affects the loss only via (a) rewards → advantages and (b) injected features, which are observations like any retrieved document. No gradient needs to flow through $\tau$, so its non-differentiability is not a compromise — it is what keeps the reward *stationary* during RL (no reward-model drift / co-evolution instability) and keeps the module auditable. The co-trained variant ($\mathcal{L} = \mathcal{L}_{GRPO} + \lambda_1\mathcal{L}_{hit} + \lambda_2\mathcal{L}_K$, with an EMA-lagged copy generating rewards) is specified as an ablation, not the default. Tradeoff acknowledged: freezing forfeits adaptation of the observation model to the policy's shifting query distribution; the EMA variant tests whether that matters.

**Anti-hacking analysis of $R_{cov}$.** (a) *Slot spamming*: spawning slots truncates $\kappa$ upward, growing the denominator — spawning is self-penalizing unless later covered, so DECOMPOSE is only worth it when real. (b) *Redundant re-retrieval*: $p_j \le 1$ makes repeated hits geometrically worthless ($\Delta C \to 0$), unlike additive bonuses which pay forever — invariance in practice, ablation (d) demonstrates it. (c) *Topical-but-useless docs inflating $m_j$*: this is your distractor-mode failure (E5 cosine cannot separate gold from same-topic distractors) and is precisely why the observation model is a **calibrated NLI cross-encoder fitted against gold hit labels**, not raw cosine — with a hard go/no-go AUC gate before any RL (Section 9).

---

## 4. Feasibility on one H200 / A100

| Item | Cost |
|---|---|
| NLI cross-encoder (DeBERTa-v3-base 184M or MiniLM 22M, fp16) | 10 docs × ≤5 slots = ≤50 pairs/turn, one batch ≈ 15–25 ms — noise vs ~1–3 s of LLM decoding |
| $p_\psi(K|q)$ MLP on frozen E5 | ~1M params, <1 ms |
| $f_1/f_0$, $\hat\pi_a$ | histograms / a few dozen scalars |
| Belief filter | closed-form, <0.1 ms |
| New VRAM | ≤0.5 GB (NLI) beside vLLM |
| GRPO | verl or OpenRLHF; LoRA r=64 on R3-RAG-Qwen; G=8 rollouts/prompt; 8k HotpotQA-train prompts/epoch ≈ 64k trajectories ≈ ~160k LLM turns — days-scale on 1×H200 with vLLM rollouts; full-param GRPO possible but not required for the claim |

Training-stability risks and their controls: mixed dense/sparse reward scales → per-component reward normalization and $\eta \le 0.3$ so $\sum R_{cov} \le \eta < R_{ans}$; length collapse / query babbling → KL anchor + $c_r$; reward hacking → Section 3.4 + monitor avg turns and coverage-vs-EM correlation during training.

---

## 5. Novelty and positioning (mid-2026 literature)

| Method (2025–26) | Tracks what | Uncertainty? | Action-conditioned belief? | Reward role | Invariance-aware? | Gold-free at test? | Trained policy? |
|---|---|---|---|---|---|---|---|
| R3-RAG (2505.23794) | per-step doc relevance | no | no | heuristic dense | no | yes | yes |
| GlobalRAG (2510.20548) | subgoal completion | no | no | additive bonus | no | yes | yes |
| HiPRAG (2510.07794) | over/under-search steps | no | no | hierarchical bonus | no | yes | yes |
| EviNote-RAG (2509.00877) / Sufficiency-to-Reflection (2507.22716) | note/sufficiency quality | no | no | additive bonus | no | yes | yes |
| CGDP (2605.07042) | predicate belief state | verbal only | no | none (training-free harness) | n/a | yes | no |
| PABU (2602.09138) | task progress (context compression) | no | no | none (SFT-style) | n/a | yes | yes |
| TASR (2606.13814) / CalVerT (2606.21777) | stop signals / verifier telemetry | calibrated scalar | no | none (inference-time) | n/a | yes | no |
| BridgeRAG (2604.03384) | bridge conditioning | no | no | none (training-free) | n/a | yes | no |
| **ACEC-Belief RAG (ours)** | **per-slot coverage posterior + $K$ posterior** | **full posterior** | **yes (routing + likelihoods)** | **potential-based shaping** | **yes (Ng et al. invariance)** | **yes** | **yes (turn-level GRPO)** |

The through-line of the reward-side literature is that "evidence sufficiency" is an *additive reward heuristic*; the belief-side literature (CGDP, PABU, Agent-BRACE) tracks state but never closes the loop into policy optimization. ACEC occupies the intersection: **the sufficiency signal is a calibrated Bayesian posterior that is simultaneously state feature, stopping criterion, and policy-invariant dense reward**, with the action-indexed observation model making per-turn credit well-defined. DIVER (2509.26209) establishes PBRS-with-GRPO as sound methodology (for diversity intrinsic rewards); ACEC contributes the *potential* — a belief functional with a train/test-asymmetric estimation story — not the shaping trick itself. Positioning must be honest on this point.

**Contribution statement (abstract-ready).** *We introduce ACEC-Belief RAG, which replaces scalar retrieval-quality beliefs in agentic multi-hop RAG with an action-conditioned Bayesian posterior over evidence coverage: which of the question's required evidence slots are satisfied, and how many exist. The posterior is computed by a monotone filter whose observation model is calibrated offline on supporting-fact annotations yet requires no gold labels at inference, and its expectation serves simultaneously as policy feature, stopping criterion, and — as a potential function — a policy-invariant dense reward that densifies GRPO's sparse outcome signal without altering the optimal policy. On HotpotQA, 2WikiMultiHopQA, MuSiQue and Bamboogle, coverage-shaped turn-level GRPO improves EM by X while reducing retrieval calls by Y, and the calibrated posterior predicts evidence completeness with Z AUC out of domain.*

---

## 6. Ablation plan — isolating each claim

| # | Ablation | Isolates | Expected signature if the claim is real |
|---|---|---|---|
| a | Pooled updates: $f^{\mathsf{tgt}}{=}f^{\mathsf{inc}}$, single $\pi$, $K{\equiv}1$ | action-conditioning (Gap A) | worse slot attribution → less-targeted rewrites, EM drop, coverage AUC drop |
| b | Gold-SF process reward only, no belief anywhere (= Phase-2 plan / GlobalRAG-style) | the belief itself vs reward heuristic | matches ACEC on labeled in-domain; loses on unlabeled-data training and Bamboogle; no stopping gate |
| c | Trajectory-level vs turn-level advantages, same rewards | the Section 3.2 analysis | trajectory-level ≈ final-coverage-bonus only; turn-level converges faster / higher |
| d | $R_{cov}$ as additive bonus $+\eta C_t$ per turn (non-potential) | invariance in practice | retrieval bloat / coverage farming: avg turns ↑ with flat EM |
| e | Belief features to policy: on / off (reward always on) | the feature channel (your prior null result) | small or zero delta — fine; the claim rests on the reward channel |
| f | $K$: oracle / predicted / fixed-2 | $K$ posterior | fixed-2 hurts MuSiQue (K up to 4), oracle bounds the gap |
| g | Observation model: calibrated NLI vs raw E5 cosine | your distractor negative result, now a motivated ablation | E5 variant's coverage AUC collapses on distractor-style docs |

---

## 7. Experiment plan

**Training.** HotpotQA train (90k with SF; your prep script exists). RL prompt set: 8–16k questions, G=8, turn-level GRPO, hybrid reward. Optional second stage: +8k SF-free prompts (NQ / synthetic multi-hop) with belief-only reward — the scaling claim.

**Evaluation.** HotpotQA dev (7405, your standing benchmark), 2WikiMultiHopQA dev, MuSiQue-Ans dev, Bamboogle (125, fully OOD, no SF ever — the generalization test for the coverage estimator). Fixed num_search=5, docs=10 to stay comparable with your tables.

**Metrics.** EM/F1 (proc), LLM-Judge (Qwen2.5-7B judge, your pipeline); efficiency: avg retrieval calls, avg docs, over-search rate (retrievals after coverage was already complete per gold) and under-search rate (stopped with gold coverage < 1); belief quality: per-slot hit AUC, coverage ECE, corr($C_T$, correctness); **accuracy-vs-retrieval-calls Pareto curves** (sweep $c_r$, and the VOI gate threshold at inference) — this is the figure that sells the stopping story.

**Baselines.** (1) R3-RAG-Qwen frozen (EM 43.2 — your table); (2) outcome-only GRPO (sparse $R_{ans}$, the cold-start pain point); (3) gold-SF process-reward GRPO, no belief (= ablation b, stands in for GlobalRAG/HiPRAG-class rewards under identical infra); (4) ACEC full. Inference-time-only comparisons: your Condition-A threshold gate and a TASR-style training-free stop on the same frozen policy vs the VOI gate. Report published numbers of Search-R1/GlobalRAG-class systems for context, clearly marked non-comparable infra.

**Decision thresholds.** The paper is viable if: ACEC > (2) by ≥ +2 EM on HotpotQA with fewer or equal retrieval calls, AND ACEC ≥ (3) in-domain while beating it on Bamboogle or on the unlabeled-data stage, AND ablations (a)/(c)/(d) show the predicted signatures. If ACEC ≈ (3) everywhere and unlabeled-data training doesn't move, the honest fallback is a shorter analysis-paper framing: "process rewards for RAG are potential-based shaping; here is the theory and the calibrated-belief generalization" — still publishable, at Findings tier.

---

## 8. Risks and failure modes, stated plainly

1. **The strongest known result may not need the belief.** Ablation (b) — gold process reward, no belief — is the most dangerous row in the paper. Your +27.9pt bridge-injection diagnostic says targeting is where the value is; gold-SF rewards also teach targeting. The belief's unique surplus is confined to: unlabeled-data scaling, OOD stopping, semantic (non-title-match) coverage, and the inference-time gate. If those four don't materialize, novelty narrows to the formal unification. Mitigation: run (b) *first*, in week 3–4, before investing in polish.
2. **Slot identification error is irreducible.** Bridge questions hide slot 2 until slot 1 resolves; the labeler and decomposer will sometimes fragment or merge slots. Coverage under-estimation early in the episode is benign (it says "continue," which is correct); *mis-binding* (marking the wrong slot covered) is the harmful mode. Measured by slot-alignment accuracy against SF; reported, not hidden. Comparison-type questions (independent slots) are the easy case; open-form non-decomposable questions degrade to $K{=}1$, where $C_t$ reduces to an answerability score — scope the claims to entity-bridge/comparison multi-hop.
3. **NLI calibration shift OOD.** $f_1/f_0$ fitted on HotpotQA may miscalibrate on Bamboogle/MuSiQue. PBRS invariance means training rewards can't be corrupted into a wrong optimum, but the *gate and features* degrade. Report ECE per dataset; recalibrate on 2Wiki as a cheap robustness check.
4. **Step-0 parametric answers** (8.5% of your data): the model answers correctly with zero retrieval, coverage stays ~0, and the shaped return correctly nets ≈ 0 extra — but the *feature* channel would say "don't answer." The policy learns to overrule it via $R_{ans}$; monitor that step-0 answering doesn't degrade.
5. **GRPO instability from mixed reward scales / variable-length groups.** Controls in Section 4; also warm-start from your SFT checkpoint and freeze $\psi$ (stationary reward) — the design's conservatism here is deliberate.
6. **The known negative result on prompt injection.** Belief prefixes hurt the frozen policy (−1~2pt). Under training the policy can learn to use them, but the design never bets on it: ablation (e) may legitimately come out null, and the paper's claims survive on the reward channel alone.
7. **SF label noise.** HotpotQA SF has known incompleteness; gold "hits" are slightly wrong. The belief reward is *less* exposed (semantic entailment vs title match) — worth one analysis paragraph, and it strengthens (iii) in Section 3.3.

---

## 9. First two weeks (one undergrad + PhD mentor; replay-first, GPU-light)

**Week 1 — observation model + filter, entirely on logged data (no new inference).**
- D1–2: runtime slot extraction + labeler $\lambda$ over the existing 7405-sample baseline `records.jsonl`; gold hit labels via SF-title Hungarian matching (reuses Phase-2 `r_sf` code). Deliverable: per-turn (slot, action-mode, NLI score, gold-hit) table.
- D3–4: fit $f_1/f_0$ (by role × bound), $\hat\pi_a$ per mode, $p_\psi(K|q)$. **Go/no-go gate: per-slot hit AUC ≥ 0.80 and K-accuracy ≥ 75% on held-out dev.** Below that, the observation model can't support the belief and the design stops here — cheaply.
- D5–7: implement Algorithm 1 + unit tests; replay all 7405 records through the filter. Deliverables: corr($C_T$, correctness), coverage-vs-EM curve, ECE; oracle headroom numbers (stop-at-gold-coverage efficiency bound; posterior-targeted rewrite vs your +27.9pt oracle injection).

**Week 2 — inference-time probes + GRPO scaffold.**
- D8–10: VOI gate + posterior-driven bridge-entity rewriting on a 1k dev subset with live inference. This is the cheap headline probe: how much of the +27.9pt oracle does the belief recover with zero training?
- D11–14: verl (or OpenRLHF) GRPO scaffold: turn-level advantage implementation, hybrid reward plumbing, LoRA r=64, smoke run (100 steps, 5k prompts, G=8). Deliverables: reward curves for outcome-only vs +$R_{cov}$ cold start (the sparsity claim, visualized), sanity dashboards (avg turns, coverage farming check). End-of-week decision memo with the mentor: proceed to full runs / re-scope per Section 7's thresholds.

---

## References (positioning set)

R3-RAG: arXiv:2505.23794 · GlobalRAG: arXiv:2510.20548 · HiPRAG: arXiv:2510.07794 · EviNote-RAG: arXiv:2509.00877 · Sufficiency-to-Reflection: arXiv:2507.22716 · Process-vs-Outcome for agentic RAG: arXiv:2505.14069 · CGDP (POMDP agentic search): arXiv:2605.07042 · PABU: arXiv:2602.09138 · TASR: arXiv:2606.13814 · CalVerT: arXiv:2606.21777 · BridgeRAG: arXiv:2604.03384 · DIVER (PBRS+GRPO): arXiv:2509.26209 · GTPO (turn-level credit): arXiv:2511.14846 · Ng, Harada, Russell (1999), *Policy invariance under reward transformations* (ICML).




