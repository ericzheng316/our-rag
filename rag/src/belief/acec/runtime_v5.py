"""Runtime belief update for ACEC v5 marginal-evidence calibration.

This module is additive: the v1-v4 runtime remains untouched.  V5 differs
from the older hit-label models because its calibrated observation is the
conditional marginal utility of the exact max-NLI-selected document.  The
runtime must therefore reproduce the offline novelty feature using only
documents selected on earlier turns of the same rollout.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .action_labeler import ActionLabel, ActionLabeler, ActionMode, Embedder
from .belief_state import NLIScorer, TurnResult
from .calibration_v5 import MarginalUtilityObservationModel
from .config import ACECConfig
from .coverage_belief import CoverageBelief, CoverageFeatures
from .docs import doc_title
from .entity_binding import augment_hypothesis_with_bound_entities
from .evidence_standard_v5 import document_novelty_score
from .hypothesis import query_to_hypothesis
from .k_posterior import KPredictor, UniformKPredictor


class RuntimeDiagnosticsV5:
    """Cheap process-level counters for smoke-run validity checks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turns = 0
        self.observations = 0
        self.bindings = 0
        self.action_counts: Counter[str] = Counter()
        self.support_sum = 0.0
        self.novelty_sum = 0.0
        self.posterior_sum = 0.0
        self.delta_coverage_sum = 0.0
        self.exact_repeat_observations = 0
        self.low_novelty_observations = 0

    def observe_turn(
        self,
        action_mode: str,
        support_scores: Sequence[float],
        novelty_scores: Sequence[float],
        gain_posteriors: Sequence[float],
        delta_coverage: float,
        new_bindings: int,
    ) -> None:
        self.turns += 1
        self.action_counts[action_mode] += 1
        self.delta_coverage_sum += float(delta_coverage)
        self.bindings += int(new_bindings)
        for support, novelty, posterior in zip(
            support_scores, novelty_scores, gain_posteriors
        ):
            self.observations += 1
            self.support_sum += float(support)
            self.novelty_sum += float(novelty)
            self.posterior_sum += float(posterior)
            if novelty <= 1e-6:
                self.exact_repeat_observations += 1
            if novelty < 0.5:
                self.low_novelty_observations += 1

    def summary(self) -> Dict[str, Any]:
        observations = max(self.observations, 1)
        turns = max(self.turns, 1)
        return {
            "turns": self.turns,
            "observations": self.observations,
            "action_counts": dict(sorted(self.action_counts.items())),
            "mean_support": self.support_sum / observations,
            "mean_novelty": self.novelty_sum / observations,
            "mean_gain_posterior": self.posterior_sum / observations,
            "mean_delta_coverage": self.delta_coverage_sum / turns,
            "exact_repeat_fraction": self.exact_repeat_observations / observations,
            "low_novelty_fraction": self.low_novelty_observations / observations,
            "bindings": self.bindings,
        }


RUNTIME_DIAGNOSTICS_V5 = RuntimeDiagnosticsV5()


class CoverageBeliefV5(CoverageBelief):
    """Coverage filter whose observations are v5 marginal-gain posteriors."""

    def __init__(
        self,
        config: ACECConfig,
        k_predictor: KPredictor,
        obs_model: MarginalUtilityObservationModel,
    ) -> None:
        super().__init__(config=config, k_predictor=k_predictor, obs_model=obs_model)
        self.obs_model = obs_model
        self.last_gain_posteriors: Dict[int, float] = {}
        self.last_novelty_scores: Dict[int, float] = {}

    def step(
        self,
        action: ActionLabel,
        slot_scores: Dict[int, float],
        slot_novelty_scores: Dict[int, float],
    ) -> CoverageFeatures:
        self.turn += 1
        mode = action.mode.value if isinstance(action.mode, ActionMode) else str(action.mode)
        beta = self.hit_betas.get(mode)
        pi_a = beta.mean if beta is not None else 0.5
        self.last_gain_posteriors = {}
        self.last_novelty_scores = {}

        for slot_index, support_score in slot_scores.items():
            if slot_index >= len(self.p):
                continue
            if slot_index not in slot_novelty_scores:
                raise ValueError(
                    f"v5 runtime is missing novelty for scored slot {slot_index}"
                )
            novelty_score = float(slot_novelty_scores[slot_index])
            role = "tgt" if slot_index == action.target_slot else "inc"
            bound = self.slots[slot_index].bound
            gain = float(
                self.obs_model.gain_posterior(
                    float(support_score), novelty_score, pi_a, role, bound
                )
            )
            if not math.isfinite(gain):
                raise ValueError("v5 observation model returned a non-finite posterior")
            gain = min(max(gain, 0.0), 1.0)
            self.last_gain_posteriors[slot_index] = gain
            self.last_novelty_scores[slot_index] = novelty_score
            self.p[slot_index] = 1.0 - (1.0 - self.p[slot_index]) * (1.0 - gain)
            if role == "tgt" and beta is not None:
                beta.update(gain)

        return self.features()


class ACECBeliefStateV5:
    """End-to-end ACEC state matching the v5 offline replay definition."""

    def __init__(
        self,
        embedder: Embedder,
        nli_scorer: NLIScorer,
        observation_model: MarginalUtilityObservationModel,
        config: Optional[ACECConfig] = None,
        k_predictor: Optional[KPredictor] = None,
        diagnostics: RuntimeDiagnosticsV5 = RUNTIME_DIAGNOSTICS_V5,
    ) -> None:
        self.config = config or ACECConfig()
        self.nli_scorer = nli_scorer
        self.labeler = ActionLabeler(
            embedder,
            tau_new=self.config.tau_new,
            tau_para=self.config.tau_para,
        )
        predictor = k_predictor or UniformKPredictor(self.config.k_max)
        self.coverage_belief = CoverageBeliefV5(
            self.config, predictor, observation_model
        )
        self._prior_selected_documents: List[str] = []
        self._diagnostics = diagnostics

    def reset(self, question: str = "") -> None:
        self.coverage_belief.reset(question)
        self.labeler.reset()
        self._prior_selected_documents = []

    def turn(
        self,
        query: Optional[str],
        new_docs: Sequence[Dict[str, Any]],
        is_answer: bool = False,
    ) -> TurnResult:
        belief = self.coverage_belief
        coverage_before = belief.coverage()

        if is_answer:
            action = ActionLabel(ActionMode.ANSWER, None)
            self._diagnostics.observe_turn(
                ActionMode.ANSWER.value, [], [], [], 0.0, 0
            )
            return TurnResult(
                features=belief.features(),
                coverage_before=coverage_before,
                coverage_after=coverage_before,
                delta_coverage=0.0,
                action=action,
                suggested_target_slot=None,
                stop_voi=True,
                slot_scores={},
                slot_best_docs={},
            )

        slot_hypotheses = [slot.hypothesis for slot in belief.slots]
        action = self.labeler.label(query or "", slot_hypotheses)
        label_max_sim = self.labeler.last_max_sim
        if action.mode == ActionMode.DECOMPOSE:
            hypothesis = (
                query_to_hypothesis(query)
                if query
                else f"unresolved sub-question {len(belief.slots) + 1}"
            )
            new_index = belief.spawn_slot(hypothesis=hypothesis)
            self.labeler.register_new_slot_query(new_index, query or "")
            action = ActionLabel(ActionMode.DECOMPOSE, new_index)

        document_entries = [document for document in new_docs if document.get("contents")]
        slot_scores: Dict[int, float] = {}
        slot_best_docs: Dict[int, Dict[str, Any]] = {}
        slot_novelty_scores: Dict[int, float] = {}
        for slot_index, slot in enumerate(belief.slots):
            if not document_entries:
                continue
            other_bound_entities = [
                entity
                for other_index, other in enumerate(belief.slots)
                if other_index != slot_index and other.bound
                for entity in other.bound_entities
            ]
            augmented_hypothesis = augment_hypothesis_with_bound_entities(
                slot.hypothesis, other_bound_entities
            )
            scored_documents = [
                (
                    self.nli_scorer.score(
                        document["contents"], augmented_hypothesis
                    ),
                    document,
                )
                for document in document_entries
            ]
            support_score, best_document = max(
                scored_documents, key=lambda pair: pair[0]
            )
            selected_text = str(best_document["contents"])
            slot_scores[slot_index] = float(support_score)
            slot_best_docs[slot_index] = best_document
            slot_novelty_scores[slot_index] = document_novelty_score(
                selected_text, self._prior_selected_documents
            )

        features = belief.step(action, slot_scores, slot_novelty_scores)
        coverage_after = belief.coverage()
        for slot_index in sorted(slot_best_docs):
            selected_text = str(slot_best_docs[slot_index]["contents"])
            if selected_text and selected_text not in self._prior_selected_documents:
                self._prior_selected_documents.append(selected_text)

        bindings_before = sum(slot.bound for slot in belief.slots)
        for slot_index, slot in enumerate(belief.slots):
            if slot.bound or slot_index not in slot_best_docs:
                continue
            if belief.p[slot_index] >= self.config.bound_threshold:
                entity = doc_title(slot_best_docs[slot_index])
                if entity:
                    belief.bind_slot(slot_index, entity)
        bindings_after = sum(slot.bound for slot in belief.slots)
        delta_coverage = coverage_after - coverage_before
        ordered_indices = sorted(belief.last_gain_posteriors)
        self._diagnostics.observe_turn(
            action.mode.value,
            [slot_scores[index] for index in ordered_indices],
            [belief.last_novelty_scores[index] for index in ordered_indices],
            [belief.last_gain_posteriors[index] for index in ordered_indices],
            delta_coverage,
            bindings_after - bindings_before,
        )

        return TurnResult(
            features=features,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            delta_coverage=delta_coverage,
            action=action,
            suggested_target_slot=belief.suggest_target_slot(),
            stop_voi=belief.should_stop_voi(),
            slot_scores=slot_scores,
            slot_best_docs=slot_best_docs,
            label_max_sim=label_max_sim,
        )

    def reward(self, turn_result: TurnResult, answer_reward: float = 0.0) -> float:
        is_answer = turn_result.action.mode == ActionMode.ANSWER
        coverage_reward = self.config.eta * (
            turn_result.coverage_after - turn_result.coverage_before
        )
        efficiency = 0.0 if is_answer else -self.config.c_r
        return (answer_reward if is_answer else 0.0) + coverage_reward + efficiency

    def bind_slot(self, index: int, entity: str) -> None:
        self.coverage_belief.bind_slot(index, entity)


__all__ = [
    "ACECBeliefStateV5",
    "CoverageBeliefV5",
    "RUNTIME_DIAGNOSTICS_V5",
    "RuntimeDiagnosticsV5",
]
