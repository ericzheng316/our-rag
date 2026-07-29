"""One-pass ACEC v7 runtime: route, score, select, then update live belief.

The implementation intentionally does not call ``ACECBeliefState.turn`` twice.
Action routing mutates the labeler's query history, so a probe-then-update
implementation would silently change REWRITE into EXPAND.  V7 performs one
transaction and asserts that set selection itself did not mutate the belief.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .action_labeler import ActionLabel, ActionMode
from .belief_simulator_v7 import BeliefSimulatorV7
from .belief_state import TurnResult
from .contracts_v7 import CandidateV7, SelectionTraceV7, SelectorStateV7
from .docs import doc_title
from .entity_binding import augment_hypothesis_with_bound_entities
from .hypothesis import query_to_hypothesis
from .set_selector_v7 import SequentialSetSelectorV7


@dataclass(frozen=True)
class V7TurnResult:
    belief_result: TurnResult
    selector_state: SelectorStateV7
    candidate_documents: Tuple[Mapping[str, Any], ...]
    selected_documents: Tuple[Mapping[str, Any], ...]
    candidate_features: Tuple[CandidateV7, ...]
    selection_trace: SelectionTraceV7
    simulator_coverage_after: float
    simulator_live_coverage_error: float
    candidate_scoring_seconds: float
    selection_seconds: float


def _candidate_id(document: Mapping[str, Any], rank: int) -> str:
    value = document.get("id")
    if value is not None and str(value).strip():
        return str(value)
    digest = hashlib.sha256(
        str(document.get("contents") or "").encode("utf-8")
    ).hexdigest()[:20]
    return f"content:{digest}:{rank}"


def _retrieval_score(document: Mapping[str, Any], rank: int) -> float:
    for key in ("retrieval_score", "score", "similarity"):
        value = document.get(key)
        if value is not None:
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                return score
    distance = document.get("distance")
    if distance is not None:
        try:
            score = -float(distance)
        except (TypeError, ValueError):
            score = math.nan
        if math.isfinite(score):
            return score
    return 1.0 / float(rank + 1)


def _slot_weights(coverage_belief: Any) -> Tuple[float, ...]:
    slot_count = len(coverage_belief.p)
    if slot_count == 0:
        return ()
    if coverage_belief.kappa is None:
        return (1.0 / slot_count,) * slot_count
    weights = []
    for slot_index in range(slot_count):
        weight = 0.0
        for k in range(slot_index + 1, coverage_belief.config.k_max + 1):
            weight += float(coverage_belief.kappa.kappa[k - 1]) / float(k)
        weights.append(weight)
    return tuple(weights)


def _score_pairs(nli_scorer: Any, pairs: Sequence[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []
    batch_method = getattr(nli_scorer, "score_batch", None)
    if callable(batch_method):
        values = batch_method(pairs)
        if len(values) != len(pairs):
            raise ValueError("NLI score_batch returned the wrong number of values")
        return [float(value) for value in values]
    return [float(nli_scorer.score(premise, hypothesis)) for premise, hypothesis in pairs]


def _score_triplets(
    nli_scorer: Any, pairs: Sequence[Tuple[str, str]]
) -> List[Tuple[float, float, float]]:
    """Return (contradiction, entailment, neutral) without changing v6 scorer."""

    if not pairs:
        return []
    triplet_method = getattr(nli_scorer, "score_triplets", None)
    if callable(triplet_method):
        rows = triplet_method(pairs)
        if len(rows) != len(pairs):
            raise ValueError("NLI score_triplets returned the wrong row count")
        return [
            tuple(float(value) for value in row)
            for row in rows
        ]
    cross_encoder = getattr(nli_scorer, "model", None)
    predict = getattr(cross_encoder, "predict", None)
    if callable(predict):
        probabilities = predict(list(pairs), apply_softmax=True)
        if len(probabilities) == len(pairs) and all(
            len(row) >= 3 for row in probabilities
        ):
            return [
                (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                )
                for row in probabilities
            ]
    entailment = _score_pairs(nli_scorer, pairs)
    return [
        (0.0, float(value), max(1.0 - float(value), 0.0))
        for value in entailment
    ]


def _structural_belief_snapshot(coverage_belief: Any) -> Tuple[Any, ...]:
    return (
        int(coverage_belief.turn),
        tuple(float(value) for value in coverage_belief.p),
        tuple(
            (
                slot.hypothesis,
                bool(slot.bound),
                tuple(slot.bound_entities),
            )
            for slot in coverage_belief.slots
        ),
        tuple(
            (key, float(beta.alpha), float(beta.beta))
            for key, beta in sorted(coverage_belief.hit_betas.items())
        ),
    )


class BeliefSelectorRuntimeV7:
    def __init__(
        self,
        selector: SequentialSetSelectorV7,
        *,
        selected_k: int = 5,
        max_turns: int = 5,
        simulator_tolerance: float = 1e-6,
    ) -> None:
        if selected_k <= 0 or max_turns <= 0:
            raise ValueError("selected K/max turns must be positive")
        self.selector = selector
        self.selected_k = int(selected_k)
        self.max_turns = int(max_turns)
        self.simulator_tolerance = float(simulator_tolerance)

    def _route_action(self, belief: Any, query: str) -> Tuple[ActionLabel, Optional[float]]:
        cb = belief.coverage_belief
        slot_hypotheses = [slot.hypothesis for slot in cb.slots]
        action = belief.labeler.label(query, slot_hypotheses)
        label_max_similarity = belief.labeler.last_max_sim
        if action.mode == ActionMode.DECOMPOSE:
            hypothesis = (
                query_to_hypothesis(query)
                if query
                else f"unresolved sub-question {len(cb.slots) + 1}"
            )
            new_index = cb.spawn_slot(hypothesis=hypothesis)
            belief.labeler.register_new_slot_query(new_index, query)
            action = ActionLabel(ActionMode.DECOMPOSE, new_index)
        return action, label_max_similarity

    def _selector_state(
        self,
        *,
        question_id: str,
        belief: Any,
        action: ActionLabel,
        retrieval_budget_remaining: int,
        selection_capacity: int,
    ) -> SelectorStateV7:
        cb = belief.coverage_belief
        features = cb.features()
        return SelectorStateV7(
            question_id=question_id,
            turn_index=int(cb.turn),
            coverage=float(features.coverage),
            coverage_std=float(features.coverage_std),
            k_entropy=float(features.k_entropy),
            slot_probabilities=tuple(float(value) for value in cb.p),
            slot_weights=_slot_weights(cb),
            slot_bound=tuple(bool(slot.bound) for slot in cb.slots),
            target_slot=action.target_slot,
            retrieval_budget_remaining=int(retrieval_budget_remaining),
            max_turns=self.max_turns,
            metadata={
                "action_mode": action.mode.value,
                "selection_capacity": int(selection_capacity),
                "evidence_membership_supervision_used": False,
            },
        )

    def _candidates(
        self,
        *,
        belief: Any,
        action: ActionLabel,
        documents: Sequence[Mapping[str, Any]],
    ) -> Tuple[Tuple[CandidateV7, ...], Dict[str, Mapping[str, Any]]]:
        cb = belief.coverage_belief
        mode = action.mode.value
        beta = cb.hit_betas.get(mode)
        pi_action = beta.mean if beta is not None else 0.5

        hypotheses = []
        for slot_index, slot in enumerate(cb.slots):
            other_entities = [
                entity
                for other_index, other in enumerate(cb.slots)
                if other_index != slot_index and other.bound
                for entity in other.bound_entities
            ]
            hypotheses.append(
                augment_hypothesis_with_bound_entities(
                    slot.hypothesis, other_entities
                )
            )

        unique_documents: List[Tuple[int, Mapping[str, Any], str]] = []
        seen_ids = set()
        for rank, document in enumerate(documents):
            if not str(document.get("contents") or "").strip():
                continue
            candidate_id = _candidate_id(document, rank)
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            unique_documents.append((rank, document, candidate_id))

        pairs = [
            (str(document.get("contents") or ""), hypothesis)
            for _, document, _ in unique_documents
            for hypothesis in hypotheses
        ]
        flat_triplets = _score_triplets(belief.nli_scorer, pairs)
        candidates: List[CandidateV7] = []
        by_id: Dict[str, Mapping[str, Any]] = {}
        offset = 0
        for rank, document, candidate_id in unique_documents:
            triplets = flat_triplets[offset : offset + len(hypotheses)]
            entailment = tuple(
                min(max(float(value[1]), 0.0), 1.0) for value in triplets
            )
            contradiction_scores = tuple(
                min(max(float(value[0]), 0.0), 1.0) for value in triplets
            )
            neutral_scores = tuple(
                min(max(float(value[2]), 0.0), 1.0) for value in triplets
            )
            offset += len(hypotheses)
            hit_probabilities = []
            for slot_index, score in enumerate(entailment):
                role = "tgt" if slot_index == action.target_slot else "inc"
                hit_probabilities.append(
                    float(
                        cb.obs_model.hit_posterior(
                            score,
                            pi_action,
                            role=role,
                            bound=cb.slots[slot_index].bound,
                        )
                    )
                )
            contradiction = (
                document.get("slot_contradiction") or contradiction_scores
            )
            neutral = document.get("slot_neutral") or neutral_scores
            candidate = CandidateV7(
                candidate_id=candidate_id,
                contents=str(document.get("contents") or ""),
                retrieval_rank=rank,
                retrieval_score=_retrieval_score(document, rank),
                slot_entailment=entailment,
                slot_hit_probabilities=tuple(hit_probabilities),
                slot_contradiction=tuple(float(v) for v in contradiction),
                slot_neutral=tuple(float(v) for v in neutral),
                signed_answer_value=float(document.get("signed_answer_value", 0.0)),
                source_quality=float(document.get("source_quality", 0.5)),
                metadata={
                    "title": str(document.get("title") or ""),
                    "source_id": str(document.get("source_id") or ""),
                },
            )
            candidates.append(candidate)
            by_id[candidate_id] = document
        return tuple(candidates), by_id

    def turn(
        self,
        *,
        question_id: str,
        belief: Any,
        query: str,
        candidate_documents: Sequence[Mapping[str, Any]],
        retrieval_budget_remaining: int,
        sample_selector: bool = False,
        selector_seed: int = 0,
        selected_k: Optional[int] = None,
    ) -> V7TurnResult:
        effective_k = self.selected_k if selected_k is None else int(selected_k)
        if effective_k <= 0:
            raise ValueError("per-turn selected K must be positive")
        cb = belief.coverage_belief
        coverage_before_action = float(cb.coverage())
        action, label_max_similarity = self._route_action(belief, query)
        state = self._selector_state(
            question_id=question_id,
            belief=belief,
            action=action,
            retrieval_budget_remaining=retrieval_budget_remaining,
            selection_capacity=effective_k,
        )
        candidate_scoring_started = time.perf_counter()
        candidates, documents_by_id = self._candidates(
            belief=belief, action=action, documents=candidate_documents
        )
        candidate_scoring_seconds = (
            time.perf_counter() - candidate_scoring_started
        )
        snapshot_before_selection = _structural_belief_snapshot(cb)
        selection_started = time.perf_counter()
        selected_candidates, trace = self.selector.select(
            state,
            candidates,
            k=effective_k,
            sample=sample_selector,
            seed=selector_seed,
        )
        selection_seconds = time.perf_counter() - selection_started
        if _structural_belief_snapshot(cb) != snapshot_before_selection:
            raise RuntimeError("v7 selector mutated the live belief")

        simulator = BeliefSimulatorV7(state, candidates)
        simulated_after = simulator.simulate(trace.selected_ids)
        selected_documents = tuple(
            documents_by_id[candidate.candidate_id]
            for candidate in selected_candidates
        )
        slot_scores: Dict[int, float] = {}
        slot_best_documents: Dict[int, Dict[str, Any]] = {}
        for slot_index in range(len(cb.slots)):
            if not selected_candidates:
                continue
            best = max(
                selected_candidates,
                key=lambda candidate: candidate.slot_entailment[slot_index],
            )
            slot_scores[slot_index] = float(best.slot_entailment[slot_index])
            slot_best_documents[slot_index] = dict(
                documents_by_id[best.candidate_id]
            )

        features = cb.step(action, slot_scores)
        coverage_after = float(cb.coverage())
        simulator_error = abs(coverage_after - simulated_after.coverage)
        if simulator_error > self.simulator_tolerance:
            raise RuntimeError(
                "v7 simulator/live belief mismatch: "
                f"{simulator_error:.8f} > {self.simulator_tolerance:.8f}"
            )

        for slot_index, slot in enumerate(cb.slots):
            if slot.bound or slot_index not in slot_best_documents:
                continue
            if cb.p[slot_index] >= belief.config.bound_threshold:
                entity = doc_title(slot_best_documents[slot_index])
                if entity:
                    cb.bind_slot(slot_index, entity)

        result = TurnResult(
            features=features,
            coverage_before=coverage_before_action,
            coverage_after=coverage_after,
            delta_coverage=coverage_after - coverage_before_action,
            action=action,
            suggested_target_slot=cb.suggest_target_slot(),
            stop_voi=cb.should_stop_voi(),
            slot_scores=slot_scores,
            slot_best_docs=slot_best_documents,
            label_max_sim=label_max_similarity,
        )
        return V7TurnResult(
            belief_result=result,
            selector_state=state,
            candidate_documents=tuple(candidate_documents),
            selected_documents=selected_documents,
            candidate_features=candidates,
            selection_trace=trace,
            simulator_coverage_after=simulated_after.coverage,
            simulator_live_coverage_error=simulator_error,
            candidate_scoring_seconds=candidate_scoring_seconds,
            selection_seconds=selection_seconds,
        )


__all__ = ["BeliefSelectorRuntimeV7", "V7TurnResult"]
