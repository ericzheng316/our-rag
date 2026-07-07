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
    tau_new: float = 0.55
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
