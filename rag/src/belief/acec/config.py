"""
ACECConfig — tunables for Action-Conditioned Evidence-Coverage Belief RAG.

See ACEC_Belief_RAG_design.md (v1.0) Sections 1.3, 3.1, 3.4, 4 for where each
value is used. Defaults match the design doc's stated defaults where given.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ACECConfig:
    # Max number of evidence slots K the K-posterior ranges over (Section 1.1).
    k_max: int = 4

    # Labeler thresholds (Section 1.1): a query targets slot j if its cosine
    # similarity to hyp_j is >= tau_new (else the labeler emits DECOMPOSE);
    # a query is EXPAND (vs REWRITE) on its target slot if it is paraphrase-
    # similar (cosine >= tau_para) to the last query that targeted that slot.
    #
    # tau_new=0.55 (the design doc's placeholder) never once triggered
    # DECOMPOSE past the first slot on real data: E5-with-'query:'-prefix
    # cosine similarity between genuinely different sub-questions in the same
    # HotpotQA question starts around 0.66 (Week-1 pilot, n=409 routing
    # decisions, min=0.660). A sweep of 0.55/0.75/0.85/0.92/0.95 against the
    # real go/no-go gate (all passed once the entailment-index bug in
    # build_acec_calibration.py's CrossEncoderNLIScorer was also fixed) showed
    # slot identification steadily improving with higher tau_new (avg slots
    # spawned 1.00 -> 1.74 against K_true=2) at only a mild hit-AUC cost
    # (0.927 -> 0.860, both comfortably above the 0.80 gate) — 0.9 balances
    # the two. Re-sweep against your own logged trajectories before trusting
    # this on a different dataset or embedder.
    tau_new: float = 0.9
    tau_para: float = 0.85

    # Reward shaping (Section 3.1): R_cov = eta * Delta C_t, R_eff = -c_r per
    # non-answer turn. Defaults as stated in the design doc.
    eta: float = 0.3
    c_r: float = 0.05

    # Capped effective sample size for the within-episode action-indexed hit-rate
    # Beta updates (Section 1.3) — bounds how much a single episode's evidence
    # can move pi_a away from its offline-fitted prior, avoiding the posterior
    # feeding itself circularly.
    ess_cap: float = 20.0

    # p_j threshold above which a slot is considered confirmed enough to bind
    # (Algorithm 1, line 4: "bind hypotheses: substitute entities confirmed in
    # covered slots into hyp_j"). Once crossed, the title of the doc that drove
    # this slot's belief update becomes its confirmed entity, and gets injected
    # as context into every other unbound slot's hypothesis before NLI scoring
    # (see entity_binding.py) — without this, an underspecified hypothesis like
    # "the actor known for his role in X" can't tell a genuine hit from a
    # same-topic distractor about the wrong entity (Week-1 pilot's diagnosed
    # false-positive: NLI score 0.999 for a document about the wrong actor).
    bound_threshold: float = 0.7

    # Action modes that route to a target slot (ANSWER is handled separately
    # and never indexes a hit-rate Beta).
    action_modes: Tuple[str, ...] = ("EXPAND", "REWRITE", "DECOMPOSE")

    # Offline-fitted priors (alpha0, beta0) per action mode for pi_a — Section 2.2.
    # Placeholder weakly-informative defaults until offline_fit.py recalibrates
    # them from logged trajectories; REWRITE/DECOMPOSE default higher than EXPAND
    # because they inject a fresh bridge entity rather than re-probing a stale slot.
    hit_prior_alpha0: Dict[str, float] = field(
        default_factory=lambda: {"EXPAND": 1.0, "REWRITE": 2.0, "DECOMPOSE": 2.0}
    )
    hit_prior_beta0: Dict[str, float] = field(
        default_factory=lambda: {"EXPAND": 2.0, "REWRITE": 2.0, "DECOMPOSE": 1.0}
    )
