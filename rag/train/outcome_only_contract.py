"""Pure reward contract for the outcome-only GRPO ablation.

This module is intentionally independent of vLLM so the scientific contract
can be tested on CPU before an expensive GPU run starts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeOnlyReward:
    answer: float
    coverage: float
    format_error: float
    retrieval_cost: float

    def answer_reward(self, correct: bool) -> float:
        return self.answer if correct else 0.0

    def retrieval_reward(self, delta_coverage: float) -> float:
        return self.coverage * delta_coverage - self.retrieval_cost


OUTCOME_ONLY_REWARD = OutcomeOnlyReward(
    answer=1.0,
    coverage=0.0,
    format_error=0.0,
    retrieval_cost=0.0,
)


def validate_outcome_only_reward(config: OutcomeOnlyReward) -> None:
    """Reject any process signal in an outcome-only experiment."""

    expected = OUTCOME_ONLY_REWARD
    if config != expected:
        raise ValueError(
            "outcome-only requires exactly answer=1, coverage=0, "
            "format_error=0, retrieval_cost=0; "
            f"received {config}"
        )
