"""Pure control contracts for ACEC-v6 training.

The module has no torch/vLLM dependency so guard and controller behavior can
be verified before an expensive GPU process is launched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class EvidenceAnswerGuard:
    min_coverage: float = 0.50
    wrong_answer_penalty: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if self.wrong_answer_penalty < 0.0:
            raise ValueError("wrong_answer_penalty must be non-negative")

    def reward_adjustment(self, correct: bool, coverage: float) -> float:
        """Penalize only wrong answers made before evidence is sufficient.

        Correct answers are never penalized, including legitimate zero-hop
        answers.  Wrong answers after sufficient evidence remain a generator
        error and are handled by the stronger answer-token KL anchor.
        """

        if correct or float(coverage) >= self.min_coverage:
            return 0.0
        return -self.wrong_answer_penalty


@dataclass
class AdaptiveKLController:
    """One-sided KL guard that never weakens the original v5 coefficient."""

    base_coef: float = 0.01
    high_watermark: float = 0.08
    recovery_watermark: float = 0.05
    increase_factor: float = 1.25
    decrease_factor: float = 1.10
    max_coef: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.base_coef <= self.max_coef:
            raise ValueError("base_coef must be positive and <= max_coef")
        if not 0.0 <= self.recovery_watermark < self.high_watermark:
            raise ValueError("KL watermarks must satisfy 0 <= recovery < high")
        if self.increase_factor <= 1.0 or self.decrease_factor <= 1.0:
            raise ValueError("KL adjustment factors must be > 1")
        self.coef = float(self.base_coef)
        self.updates = 0

    def update(self, observed_kl: float) -> float:
        observed = max(float(observed_kl), 0.0)
        if observed > self.high_watermark:
            self.coef = min(self.max_coef, self.coef * self.increase_factor)
        elif observed < self.recovery_watermark and self.coef > self.base_coef:
            self.coef = max(self.base_coef, self.coef / self.decrease_factor)
        self.updates += 1
        return self.coef

    def snapshot(self) -> Dict[str, float]:
        payload = asdict(self)
        payload.update({"coef": self.coef, "updates": self.updates})
        return payload


def effective_kl_coefficient(
    base_coefficient: float,
    *,
    turn_kind: str,
    answer_multiplier: float,
) -> float:
    """Return the explicit per-turn KL coefficient used by ACEC-v6."""

    if base_coefficient < 0.0:
        raise ValueError("base_coefficient must be non-negative")
    if answer_multiplier < 1.0:
        raise ValueError("answer_multiplier must be >= 1")
    if turn_kind == "answer":
        return float(base_coefficient) * float(answer_multiplier)
    return float(base_coefficient)
