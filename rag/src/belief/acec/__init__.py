"""
ACEC-Belief RAG — Action-Conditioned Evidence-Coverage Belief.

Implements the design in ACEC_Belief_RAG_design.md (v1.0, 2026-07-02): an
action-conditioned Bayesian posterior over per-slot evidence coverage,
replacing the pooled system-level Beta-Bernoulli belief in
`rag.src.belief.belief_state.BeliefState` (kept as-is for ablation, per
CLAUDE.md). See that file for the legacy design; this package does not
depend on it.
"""

from .action_labeler import ActionLabel, ActionLabeler, ActionMode, Embedder
from .belief_state import ACECBeliefState, NLIScorer, TurnResult
from .config import ACECConfig
from .coverage_belief import CoverageBelief, CoverageFeatures
from .datasets import GoldEvidenceAdapter, get_adapter
from .docs import doc_title, title_from_content
from .e5_adapter import E5QueryEmbedder
from .entity_binding import augment_hypothesis_with_bound_entities
from .hypothesis import query_to_hypothesis
from .k_posterior import KPosterior, KPredictor, SplitCountKPredictor, UniformKPredictor
from .observation_model import BetaDensity, HitRateBeta, ObservationModel
from .reward import (
    efficiency_penalty,
    gold_coverage_reward,
    per_turn_reward,
    potential_shaped_reward,
    turn_level_advantages,
    turn_level_returns,
)
from .slots import Slot, SlotStore

__all__ = [
    "ACECBeliefState",
    "ACECConfig",
    "ActionLabel",
    "ActionLabeler",
    "ActionMode",
    "BetaDensity",
    "CoverageBelief",
    "CoverageFeatures",
    "E5QueryEmbedder",
    "Embedder",
    "GoldEvidenceAdapter",
    "HitRateBeta",
    "KPosterior",
    "KPredictor",
    "NLIScorer",
    "ObservationModel",
    "Slot",
    "SlotStore",
    "SplitCountKPredictor",
    "TurnResult",
    "UniformKPredictor",
    "augment_hypothesis_with_bound_entities",
    "doc_title",
    "efficiency_penalty",
    "get_adapter",
    "gold_coverage_reward",
    "per_turn_reward",
    "potential_shaped_reward",
    "query_to_hypothesis",
    "title_from_content",
    "turn_level_advantages",
    "turn_level_returns",
]
