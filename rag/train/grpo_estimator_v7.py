"""ACEC v7 query-policy estimator and coverage reward contract.

The selector is frozen during v7 query GRPO.  It is part of the environment,
so the token-level estimator remains the validated v6 estimator.  This module
adds an explicit reward-mode contract and refuses any second predicted-gain
bonus.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

class CoverageRewardModeV7(str, Enum):
    LEGACY_COVERAGE_AUX = "legacy_coverage_aux"
    STRICT_PBRS = "strict_pbrs"


def terminal_potential_correction_v7(
    coverage: float,
    eta: float,
    mode: CoverageRewardModeV7 | str,
) -> float:
    selected = CoverageRewardModeV7(mode)
    if selected == CoverageRewardModeV7.STRICT_PBRS:
        return -float(eta) * float(coverage)
    return 0.0


def grpo_loss_v7(*args: Any, predicted_gain_bonus: float = 0.0, **kwargs: Any):
    if abs(float(predicted_gain_bonus)) > 1e-12:
        raise ValueError(
            "ACEC v7 forbids a second predicted-gain reward; "
            "predicted gain is a selector feature only"
        )
    from grpo_estimator_v6 import grpo_loss_v6

    return grpo_loss_v6(*args, **kwargs)


__all__ = [
    "CoverageRewardModeV7",
    "grpo_loss_v7",
    "terminal_potential_correction_v7",
]
