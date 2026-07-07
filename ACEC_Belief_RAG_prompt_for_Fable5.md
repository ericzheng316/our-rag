# Prompt for Claude Fable 5 — Designing Action-Conditioned Evidence-Coverage Belief RAG (ACEC-Belief RAG)

> Copy everything below the line into your Fable 5 conversation as the opening message.

---

## Role and mandate

You are acting as a senior research scientist specializing in retrieval-augmented generation (RAG), Bayesian/POMDP-style reinforcement learning, and policy-gradient methods for LLM agents (PPO, GRPO). I am an undergraduate researcher extending an existing working system, and I need you to **design a new algorithm rigorously enough that it can anchor the core technical contribution of a top-tier NLP/ML venue submission** (ACL/EMNLP/NAACL main or Findings, NeurIPS/ICLR, or SIGIR).

Do not brainstorm loosely. Produce a fully specified algorithm: formal definitions, update equations, pseudocode, a training objective compatible with GRPO, and a defensible novelty argument against recent literature. Where genuine open design choices exist, enumerate 2–3 concrete alternatives, argue for a default, and state what an ablation would need to show to justify it. Think step by step and show your reasoning before committing to the final design — I would rather see you reject a weak version of the idea and propose a stronger one than force-fit the first formalization that comes to mind.

---

## 1. Background: the system this idea extends

I have a working multi-hop QA RAG system built on a **BAMDP (Bayes-Adaptive MDP)** formulation, currently used for interview prep and now being pushed toward a publishable contribution:

- **Backbone / inference:** Qwen2.5-7B served via vLLM. Retrieval corpus: ~17M Wikipedia passages (wiki18_100w-style), FAISS flat index, E5-base-v2 embeddings.
- **State representation:** a 9-dim vector = 3-dim task features `[query_len, num_docs, step]` concatenated with a 6-dim belief vector `[ret_mean, ret_unc, llm_mean, llm_unc, noise_mean, noise_unc]` derived from Beta posteriors.
- **Belief state:** independent Beta-Bernoulli conjugate posteriors over system-level competence parameters — `θ_ret` (retriever quality), `θ_llm` (generator reliability), `θ_noise` (corpus noise rate), and in some variants `θ_diff` (query difficulty). Updates are simple conjugate increments (`success → α+=1`, `fail → β+=1`) triggered by binary observations (`ret_success`, `llm_correct`, `has_noise`), **independent of which action produced the observation**.
- **Two-layer belief architecture:** (a) document-level belief `b_t(s)` — each retrieved document carries its own Beta(α,β), persisted across hops, bridged to the system-level slot beliefs via an NLI cross-encoder score; (b) system-level belief `b_t(θ)` as above.
- **Action space:** at minimum `{STOP, REWRITE, EXPAND}`; the policy decides whether/how to keep retrieving each hop.
- **Reward (current best design):** a 3-component shape — dense per-hop `R_ret` (Δθ_ret as immediate signal), an efficiency penalty `R_eff`, and a sparse final `R_ans` (answer correctness). Outcome-only reward was tried first and caused **reward sparsity and non-convergence during GRPO cold start** — this is a real pain point I want the new design to help fix.
- **Results so far:** HotpotQA EM ≈ 43.2%, LLM-Judge ≈ 62.8%.
- **Compute reality:** rented H200/A100 (Vast.ai / university cluster), solo undergraduate researcher with a PhD-student mentor, no frontier-scale training budget. Any new module should be lightweight (small MLP/GRU-scale, not a new large network) and must not blow up rollout latency.

---

## 2. The new idea to formalize: Action-Conditioned Evidence-Coverage Belief RAG

Two gaps in the current design motivate this:

**Gap A — belief updates are action-agnostic.** The Beta-Bernoulli update rule is the same regardless of *which* action (`STOP/REWRITE/EXPAND`, or which hop-slot was targeted) produced the observation. A BAMDP formally allows the transition/observation model to depend on the action, but my implementation currently pools all evidence into the same likelihood model.

**Gap B — no explicit notion of "have we covered what's needed."** `θ_ret/θ_llm/θ_noise` track *component quality/reliability*, not *coverage of the target evidence set* for a K-hop question. Stopping is currently governed by `max_steps` or `θ_ret` thresholds, not by whether the retrieved evidence set actually spans the reasoning chain needed to answer the question. This is exactly the mechanism recent work (adaptive-retrieval methods with "evidence sufficiency" stopping, lazy/redundancy penalties, process-reward RAG systems) has been converging on from a *heuristic reward-shaping* angle — I want a *formal belief-theoretic* treatment instead.

**Working definition of ACEC-Belief RAG** (please critique/refine this, don't just accept it):

1. **Action-conditioned belief transition.** Redefine the belief update as `b_{t+1} = τ(b_t, a_t, o_t)`, where the action `a_t` explicitly indexes the observation/likelihood model (e.g., a `REWRITE` on slot *i* updates different pseudo-counts than an `EXPAND`, and the *identity* of the targeted hop-slot matters). Formalize this as an action-indexed family of Beta-Bernoulli (or richer) update rules, and justify why this is more than "just add the action as a feature."
2. **Evidence-coverage belief.** Introduce a new belief component `C_t` tracking, for a (possibly unknown) K-hop question, the latent fraction/set of required evidence "slots" satisfied so far — distinct from retrieval *quality*. This should be usable both as (a) a policy input (to decide STOP vs continue) and (b) a dense reward signal (`R_cov`), directly targeting the reward-sparsity problem noted above.
3. **Coverage without gold decomposition at test time.** HotpotQA-style datasets provide `supporting_facts` annotations — usable as **weak/distant supervision for a coverage-prediction auxiliary loss during training**, but coverage must be estimated *without* gold slot labels at inference time (e.g., via the same NLI cross-encoder machinery already bridging doc-level to slot-level belief). Please design explicitly for this train/test asymmetry — I consider this the hardest and most reviewer-scrutinized part of the idea, so don't hand-wave it.

---

## 3. Hard requirement: forward AND backward compatibility with GRPO

This is the requirement I most need you to nail, and I want a dedicated section addressing it explicitly:

- **Forward:** the belief representation (action-conditioned + coverage) must be computable **online, causally**, during rollout collection — no peeking at future hops — and must produce a feature the policy can condition on at each step, plus quantities usable to construct `R_cov`.
- **Backward:** GRPO's policy gradient flows through log-probabilities of sampled actions/tokens, not through arbitrary intermediate computations. So: explicitly state (a) which parts of the belief module, if any, are *learned* parameters vs. fixed conjugate-update statistics; (b) for any learned parameters (e.g., an amortized coverage encoder), how they receive a training signal — via the GRPO policy-gradient path itself, via a separate auxiliary supervised/self-supervised loss (e.g., coverage-prediction loss against the weak `supporting_facts` labels), or via a jointly-optimized combination — and give the combined loss. Do not just describe "a Bayesian filter" and call it done; I need the actual credit-assignment story.
- Note explicitly if you recommend keeping some components **intentionally non-differentiable** (e.g., hard conjugate Beta updates) and argue why that's fine — that's a legitimate answer, but it must be a stated design decision with tradeoffs, not an oversight.

---

## 4. Recent literature to position against (as of mid-2026 — please also search for anything newer)

Use these as reference points for novelty framing, not necessarily to reproduce exactly. Distinguish ACEC-Belief RAG from: reward-shaping approaches to evidence sufficiency and redundancy penalties in agentic multi-hop RAG (e.g., lazy-stop / redundancy-penalty reward terms in graph-routed GRPO systems); process-reward and evidence-quality co-evolution methods for RAG (e.g., dense process rewards over evidence extraction quality); curriculum-scheduled multi-factor reward RAG agents; metacognitive multi-agent RAG that monitors "evidence sufficiency" via prompting rather than a formal belief model; and adaptive-RAG surveys framing retrieval budgeting as "stopping when evidence is sufficient." The throughline in this literature is that "evidence sufficiency/coverage" is being approached almost entirely as a **reward-engineering heuristic**, not as an explicit, action-conditioned Bayesian belief that is part of the state and interacts formally with the BAMDP transition model. That gap is the novelty angle I want you to sharpen or correct.

---

## 5. Deliverables — please structure your answer with these sections

1. **Problem formalization.** Full BAMDP definition: state/action/observation spaces, the action-indexed transition/observation model for `τ(b_t, a_t, o_t)`, and the coverage belief `C_t` (choose and justify a distribution family — Beta, Dirichlet-multinomial over slots, or other).
2. **Core algorithm.** Math + pseudocode for the belief update mechanism end-to-end (one rollout step).
3. **GRPO integration.** The full training objective: reward decomposition (how `R_cov` composes with `R_ret`, `R_eff`, `R_ans`), group-relative advantage computation, and the credit-assignment/differentiability story from Section 3 above.
4. **Feasibility analysis** for a 7B model on a rented H200/A100 budget — parameter count of any new module, added rollout latency, training stability risks (e.g., reward hacking on the coverage signal).
5. **Novelty & positioning.** A short table contrasting ACEC-Belief RAG against the reward-heuristic evidence-sufficiency methods above, plus a 3–4 sentence contribution statement suitable for a paper abstract.
6. **Ablation plan.** Which components to strip out to isolate the contribution of (a) action-conditioning, (b) the coverage belief itself, (c) the differentiable vs. conjugate-update choice.
7. **Experiment plan.** Datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle), metrics (EM/F1, LLM-Judge, avg. hop count / retrieval calls), baselines to compare against (my current BAMDP-RAG belief system as the primary baseline, plus 1–2 from Section 4).
8. **Risks and failure modes**, stated honestly (e.g., non-identifiability of "slots" without gold decomposition at test time, reward hacking, generalization to open-ended non-fixed-hop questions).
9. **First two weeks of concrete next steps**, scoped for one undergraduate researcher with a PhD-student mentor.

Target length: roughly 2,500–4,000 words. Use LaTeX-style math (`$...$` / `$$...$$`) and at least one clearly formatted pseudocode block. Be willing to tell me if part of the idea as stated is weak or under-specified — I'd rather you push back than flatter the premise.
