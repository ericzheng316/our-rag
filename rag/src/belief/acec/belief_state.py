"""
ACECBeliefState — end-to-end orchestrator for one rollout turn (Algorithm 1,
design doc Section 2.1). Ties together the action labeler, the per-slot NLI
scorer, and the monotone coverage filter, and exposes reward/stopping signals
for GRPO integration (Section 3) and inference-time gating (Section 2.1).

No gold labels are used here — supporting_facts only enter through the
offline calibration in offline_fit.py (Section 2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .action_labeler import ActionLabel, ActionLabeler, ActionMode, Embedder
from .config import ACECConfig
from .coverage_belief import CoverageBelief, CoverageFeatures
from .k_posterior import KPredictor, UniformKPredictor
from .observation_model import ObservationModel
from .reward import efficiency_penalty, potential_shaped_reward


class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> float: ...


@dataclass
class TurnResult:
    features: CoverageFeatures
    coverage_before: float
    coverage_after: float
    delta_coverage: float
    action: ActionLabel
    suggested_target_slot: Optional[int]
    stop_voi: bool
    # Raw per-slot NLI entailment scores m_j scored this turn (slot idx -> m_j).
    # This is the Section 2.2 Week-1 deliverable: "per-turn (slot, action-mode,
    # NLI score, gold-hit) table" — offline_fit.py needs these raw scores, not
    # just the aggregated coverage features.
    slot_scores: Dict[int, float] = field(default_factory=dict)


class ACECBeliefState:
    """
    Usage:
        belief = ACECBeliefState(embedder, nli_scorer)
        belief.reset(question)
        for turn in rollout:
            result = belief.turn(query=turn.query, new_docs=turn.docs, is_answer=turn.is_answer)
            reward = belief.reward(result, answer_reward=turn.answer_reward if turn.is_answer else 0.0)
    """

    def __init__(
        self,
        embedder: Embedder,
        nli_scorer: NLIScorer,
        config: Optional[ACECConfig] = None,
        obs_model: Optional[ObservationModel] = None,
        k_predictor: Optional[KPredictor] = None,
    ):
        self.config = config or ACECConfig()
        self.nli_scorer = nli_scorer
        self.labeler = ActionLabeler(embedder, tau_new=self.config.tau_new, tau_para=self.config.tau_para)
        self.coverage_belief = CoverageBelief(
            config=self.config,
            k_predictor=k_predictor or UniformKPredictor(self.config.k_max),
            obs_model=obs_model or ObservationModel(),
        )

    def reset(self, question: str = "") -> None:
        self.coverage_belief.reset(question)
        self.labeler.reset()

    # ------------------------------------------------------------------
    def turn(
        self,
        query: Optional[str],
        new_docs: Sequence[Dict[str, Any]],
        is_answer: bool = False,
    ) -> TurnResult:
        """One ACEC turn, executed after every policy turn (Algorithm 1)."""
        cb = self.coverage_belief
        coverage_before = cb.coverage()

        if is_answer:
            action = ActionLabel(ActionMode.ANSWER, None)
            return TurnResult(cb.features(), coverage_before, coverage_before, 0.0, action, None, True, {})

        slot_hyps = [s.hypothesis for s in cb.slots]
        action = self.labeler.label(query or "", slot_hyps)

        if action.mode == ActionMode.DECOMPOSE:
            new_idx = cb.spawn_slot(hypothesis=query or f"unresolved sub-question {len(cb.slots) + 1}")
            self.labeler.register_new_slot_query(new_idx, query or "")
            action = ActionLabel(ActionMode.DECOMPOSE, new_idx)

        doc_texts = [d.get("contents", "") for d in new_docs if d.get("contents")]
        slot_scores: Dict[int, float] = {}
        for j, slot in enumerate(cb.slots):
            if not doc_texts:
                continue
            slot_scores[j] = max(self.nli_scorer.score(text, slot.hypothesis) for text in doc_texts)

        features = cb.step(action, slot_scores)
        coverage_after = cb.coverage()

        return TurnResult(
            features=features,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            delta_coverage=coverage_after - coverage_before,
            action=action,
            suggested_target_slot=cb.suggest_target_slot(),
            stop_voi=cb.should_stop_voi(),
            slot_scores=slot_scores,
        )

    # ------------------------------------------------------------------
    def reward(self, turn_result: TurnResult, answer_reward: float = 0.0) -> float:
        """r_t = 1[t=T] R_ans + eta * Delta C_t - c_r * 1[a_t != ANSWER] (Section 3.1)."""
        is_answer = turn_result.action.mode == ActionMode.ANSWER
        r_cov = potential_shaped_reward(turn_result.coverage_before, turn_result.coverage_after, self.config.eta)
        r_eff = efficiency_penalty(is_answer, self.config.c_r)
        r_ans = answer_reward if is_answer else 0.0
        return r_ans + r_cov + r_eff

    def bind_slot(self, idx: int, entity: str) -> None:
        """Substitute a confirmed bridge entity into slot `idx`'s hypothesis
        (Algorithm 1, line 4), so its next NLI probe is entity-grounded."""
        self.coverage_belief.bind_slot(idx, entity)


# ── Smoke test (pure Python, no torch/transformers/GPU required) ──────────────
if __name__ == "__main__":
    import numpy as np

    class DummyEmbedder:
        """Deterministic bag-of-words embedder for offline testing."""

        def encode(self, texts: List[str], **kwargs) -> np.ndarray:
            vocab: Dict[str, int] = {}
            for t in texts:
                for w in t.lower().split():
                    vocab.setdefault(w, len(vocab))
            mat = np.zeros((len(texts), max(len(vocab), 1)))
            for i, t in enumerate(texts):
                for w in t.lower().split():
                    mat[i, vocab[w]] += 1.0
            return mat

    class DummyNLIScorer:
        """Word-overlap Jaccard score standing in for a cross-encoder."""

        def score(self, premise: str, hypothesis: str) -> float:
            p, h = set(premise.lower().split()), set(hypothesis.lower().split())
            if not p or not h:
                return 0.0
            return len(p & h) / len(p | h)

    belief = ACECBeliefState(DummyEmbedder(), DummyNLIScorer())
    question = "What government position did Shirley Temple hold under the president who succeeded Nixon?"
    belief.reset(question)

    print("Turn 1: DECOMPOSE (no slots yet)")
    r1 = belief.turn(
        query="Shirley Temple government position",
        new_docs=[{"contents": "Shirley Temple was appointed United States Ambassador to Ghana."}],
    )
    print(" ", r1.action, "C=%.3f -> %.3f" % (r1.coverage_before, r1.coverage_after))

    print("Turn 2: REWRITE targeting the same slot with a fresh bridge entity")
    r2 = belief.turn(
        query="United States Ambassador to Ghana appointing president",
        new_docs=[{"contents": "Shirley Temple was appointed Ambassador to Ghana by president Gerald Ford."}],
    )
    print(" ", r2.action, "C=%.3f -> %.3f" % (r2.coverage_before, r2.coverage_after))
    print("  reward:", belief.reward(r2))
    print("  suggest_target_slot:", r2.suggested_target_slot, " stop_voi:", r2.stop_voi)

    print("Turn 3: ANSWER")
    r3 = belief.turn(query=None, new_docs=[], is_answer=True)
    print("  final reward (EM=1):", belief.reward(r3, answer_reward=1.0))
    print("  features:", r3.features.to_vector(belief.config.k_max))
