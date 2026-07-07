"""
Reward decomposition as potential-based shaping — design doc Section 3.

Phi(hyperstate) := C_t. R_cov = eta * (Phi(s_{t+1}) - Phi(s_t)) is Ng-Harada-
Russell (1999) policy-invariant shaping: it leaves the optimal policy
unchanged for any Phi (including a miscalibrated one), unlike additive
sufficiency/redundancy bonuses. With gamma=1 the shaping telescopes:
sum_t R_cov,t = eta * (C_T - C_0).

Section 3.2's key subtlety: under trajectory-level GRPO advantages, the
shared-question group baseline cancels C_0 and all per-turn structure
collapses to a final-coverage bonus. Dense credit assignment requires
turn-level advantages (`turn_level_advantages` below).
"""

from __future__ import annotations

import math
from typing import List, Sequence


def potential_shaped_reward(coverage_before: float, coverage_after: float, eta: float = 0.3) -> float:
    """R_cov = eta * (Phi(s_{t+1}) - Phi(s_t)), Phi = C_t (Section 3.1)."""
    return eta * (coverage_after - coverage_before)


def efficiency_penalty(is_terminal_answer: bool, c_r: float = 0.05) -> float:
    """R_eff = -c_r * 1[a_t != ANSWER] (Section 3.1)."""
    return 0.0 if is_terminal_answer else -c_r


def per_turn_reward(
    coverage_before: float,
    coverage_after: float,
    is_terminal_answer: bool,
    answer_reward: float = 0.0,
    eta: float = 0.3,
    c_r: float = 0.05,
) -> float:
    """r_t = 1[t=T] R_ans + eta * Delta(Phi) - c_r * 1[a_t != ANSWER] (Section 3.1)."""
    r_cov = potential_shaped_reward(coverage_before, coverage_after, eta)
    r_eff = efficiency_penalty(is_terminal_answer, c_r)
    r_ans = answer_reward if is_terminal_answer else 0.0
    return r_ans + r_cov + r_eff


def gold_coverage_reward(new_sf_titles_found: int, total_sf_titles: int, eta: float = 0.3) -> float:
    """
    r^sf_t = |new SF titles found at t| / |SF| is exactly Delta(Phi*) for the
    gold-coverage potential Phi* = fraction of supporting_facts covered
    (Section 3.1's unification with the planned Phase-2 process reward).
    Use this in place of `potential_shaped_reward` on SF-labeled prompts
    (Section 3.3's hybrid reward).
    """
    if total_sf_titles <= 0:
        return 0.0
    return eta * (new_sf_titles_found / total_sf_titles)


def turn_level_returns(rewards_per_turn: Sequence[float]) -> List[float]:
    """G_{i,t} = sum_{t' >= t} r_{i,t'} for one rollout (Section 3.2)."""
    returns = [0.0] * len(rewards_per_turn)
    running = 0.0
    for t in range(len(rewards_per_turn) - 1, -1, -1):
        running += rewards_per_turn[t]
        returns[t] = running
    return returns


def turn_level_advantages(group_returns: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """
    A_{i,t} = (G_{i,t} - mu_t) / (sigma_t + eps), with mu_t/sigma_t computed
    over rollouts in the group still alive at turn t (Section 3.2). Falls back
    to the raw return when fewer than 2 rollouts remain alive at t.

    `group_returns[i]` is `turn_level_returns(rewards_per_turn_i)` for rollout
    i in a GRPO group (all sharing the same question/prompt).
    """
    max_len = max((len(r) for r in group_returns), default=0)
    advantages = [[0.0] * len(r) for r in group_returns]
    for t in range(max_len):
        alive = [(i, r[t]) for i, r in enumerate(group_returns) if len(r) > t]
        if len(alive) < 2:
            for i, val in alive:
                advantages[i][t] = val
            continue
        vals = [v for _, v in alive]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var)
        for i, v in alive:
            advantages[i][t] = (v - mu) / (sigma + eps)
    return advantages
