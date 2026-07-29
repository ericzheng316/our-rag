"""Pure, non-mutating coverage simulation for ACEC v7 set selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .contracts_v7 import CandidateV7, SelectorStateV7


def _bernoulli_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


@dataclass(frozen=True)
class SimulatedBeliefV7:
    slot_probabilities: Tuple[float, ...]
    coverage: float
    entropy: float


class BeliefSimulatorV7:
    """Simulate selected-set effects without touching the live belief.

    Candidate ``slot_hit_probabilities`` are the frozen observation model's
    action-conditioned hit posteriors.  For each slot the simulator first
    chooses the candidate with maximum raw entailment, then uses that exact
    candidate's hit posterior.  This matches ``ACECBeliefState.turn`` even if
    a fitted observation density is not globally monotone in the NLI score.
    """

    def __init__(
        self, state: SelectorStateV7, candidates: Sequence[CandidateV7]
    ) -> None:
        self.state = state
        self.candidates: Dict[str, CandidateV7] = {}
        for candidate in candidates:
            if candidate.candidate_id in self.candidates:
                raise ValueError(f"duplicate candidate id {candidate.candidate_id!r}")
            if len(candidate.slot_hit_probabilities) != len(
                state.slot_probabilities
            ):
                raise ValueError("candidate slot vector does not match selector state")
            self.candidates[candidate.candidate_id] = candidate
        self._base = self._from_hits((0.0,) * len(state.slot_probabilities))

    @property
    def base(self) -> SimulatedBeliefV7:
        return self._base

    def _coverage(self, probabilities: Sequence[float]) -> float:
        return sum(
            weight * probability
            for weight, probability in zip(
                self.state.slot_weights, probabilities
            )
        )

    def _entropy(self, probabilities: Sequence[float]) -> float:
        return sum(
            weight * _bernoulli_entropy(probability)
            for weight, probability in zip(
                self.state.slot_weights, probabilities
            )
        )

    def _from_hits(self, slot_hits: Sequence[float]) -> SimulatedBeliefV7:
        probabilities = tuple(
            1.0 - (1.0 - prior) * (1.0 - hit)
            for prior, hit in zip(self.state.slot_probabilities, slot_hits)
        )
        return SimulatedBeliefV7(
            slot_probabilities=probabilities,
            coverage=self._coverage(probabilities),
            entropy=self._entropy(probabilities),
        )

    def simulate(self, selected_ids: Iterable[str]) -> SimulatedBeliefV7:
        hits = [0.0] * len(self.state.slot_probabilities)
        best_entailment = [-math.inf] * len(self.state.slot_probabilities)
        for candidate_id in selected_ids:
            try:
                candidate = self.candidates[str(candidate_id)]
            except KeyError as exc:
                raise ValueError(
                    f"candidate {candidate_id!r} is outside the simulator pool"
                ) from exc
            for index, value in enumerate(candidate.slot_hit_probabilities):
                entailment = float(candidate.slot_entailment[index])
                if entailment > best_entailment[index]:
                    best_entailment[index] = entailment
                    hits[index] = float(value)
        return self._from_hits(hits)

    def conditional_metrics(
        self, candidate_id: str, selected_ids: Sequence[str]
    ) -> Mapping[str, float]:
        before = self.simulate(selected_ids)
        if candidate_id in selected_ids:
            return {
                "conditional_coverage_gain": 0.0,
                "information_gain": 0.0,
                "coverage_after": before.coverage,
                "entropy_after": before.entropy,
            }
        after = self.simulate((*selected_ids, candidate_id))
        return {
            "conditional_coverage_gain": after.coverage - before.coverage,
            "information_gain": before.entropy - after.entropy,
            "coverage_after": after.coverage,
            "entropy_after": after.entropy,
        }


__all__ = ["BeliefSimulatorV7", "SimulatedBeliefV7"]
