"""ACEC v8 lookahead math core — the owner's redesign, math layer only.

Implements the charter's decision loop primitives, deliberately free of any
model/IO dependency so every property is offline-testable in milliseconds:

- slot posteriors over binding (Bayes odds updates, evidence-calibrated);
- three entity-directed actions with one-step expected return E[R(s')]
  from executed-branch judge scores (greedy determinism => n=1 is exact);
- an action prior P(a | state features) updated per training round
  (softmax-linear, REINFORCE-style update) serving as the low-budget policy
  and exploration fallback;
- epsilon-greedy / UCB selection over branch values;
- the lookahead budget arithmetic used by the cost-matched comparison.

Everything here feeds `door`-style offline gates before touching the
pipeline; nothing imports torch, vllm or the observer stack.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence


class ActionV8(str, Enum):
    QUERY_TARGET = "query_target"   # 沿当前目标槽实体继续查
    PIVOT_NEXT = "pivot_next"       # 换下一个槽实体查
    ANSWER = "answer"               # 现在作答

    @classmethod
    def all(cls) -> List["ActionV8"]:
        return [cls.QUERY_TARGET, cls.PIVOT_NEXT, cls.ANSWER]


def _clamp01(p: float, eps: float = 1e-6) -> float:
    return min(max(p, eps), 1.0 - eps)


def _logit(p: float) -> float:
    p = _clamp01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── 槽后验 ──────────────────────────────────────────────────────────────────

@dataclass
class SlotV8:
    """One evidence slot: a (possibly placeholder) entity and its binding belief."""

    description: str
    entity: Optional[str] = None      # None == 占位符（bridge 实体待绑定）
    p_bound: float = 0.2              # 先验 ≈ 实测 belief 绑定率量级

    def update(self, evidence_score: float, weight: float = 1.0) -> float:
        """Bayes odds update from a calibrated evidence score in (0,1).

        score 0.5 is uninformative; weight scales the log-likelihood ratio
        (weight=0 disables the update). Returns the new posterior.
        """
        llr = weight * _logit(evidence_score)
        self.p_bound = _clamp01(_sigmoid(_logit(self.p_bound) + llr))
        return self.p_bound

    def bind(self, entity: str, p: float) -> None:
        self.entity = entity
        self.p_bound = _clamp01(p)


# ── 一步期望与动作价值 ──────────────────────────────────────────────────────

def expected_return(branch_scores: Sequence[float]) -> float:
    """E[R(s')] estimate from executed-branch judge scores.

    n=1 is exact only conditional on (a) a deterministic stack — measured to
    be ~99% true under greedy decoding (vLLM batching flips ~1% of greedy
    trajectories; mechC none-arms differed by ±0.015 mean retrievals) — and
    (b) the value being defined under the greedy *continuation* policy
    (rollout policy improvement), not V*. Use n>=2 under sampling, and pair
    n=1 greedy use with ``select_action(tie_margin=...)`` so decisions inside
    the noise floor defer to the prior instead of pretending precision.
    """
    if not branch_scores:
        raise ValueError("expected_return requires at least one branch score")
    return sum(branch_scores) / len(branch_scores)


def action_values(
    branch_returns: Mapping[ActionV8, float],
    costs: Optional[Mapping[ActionV8, float]] = None,
) -> Dict[ActionV8, float]:
    """Q(s,a) = E[R(s'_a)] - cost(a)."""
    costs = costs or {}
    return {a: float(r) - float(costs.get(a, 0.0)) for a, r in branch_returns.items()}


# ── 动作先验（按训练轮次更新）────────────────────────────────────────────────

@dataclass
class ActionPriorV8:
    """Softmax-linear P(a | features), REINFORCE-style per-round updates.

    Low-budget policy + exploration prior; the lookahead overrides it when
    branch values are available.
    """

    feature_names: Sequence[str]
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bias: Dict[str, float] = field(default_factory=dict)
    # 硬探索地板：对外概率 = (1-τ)·softmax + τ/K，最小概率恒 ≥ τ/K。
    # 熵正则只能延缓持续同向更新下的塌缩（平衡点随 advantage/β 走），
    # 混合地板才是保证；ε-greedy 是第三层独立保险。
    uniform_mix: float = 0.06

    def __post_init__(self) -> None:
        for a in ActionV8.all():
            self.weights.setdefault(a.value, {f: 0.0 for f in self.feature_names})
            self.bias.setdefault(a.value, 0.0)

    def _logits(self, features: Mapping[str, float]) -> Dict[str, float]:
        out = {}
        for a in ActionV8.all():
            w = self.weights[a.value]
            out[a.value] = self.bias[a.value] + sum(
                w[f] * float(features.get(f, 0.0)) for f in self.feature_names
            )
        return out

    def _softmax(self, features: Mapping[str, float]) -> Dict[ActionV8, float]:
        logits = self._logits(features)
        m = max(logits.values())
        exps = {a: math.exp(v - m) for a, v in logits.items()}
        z = sum(exps.values())
        return {ActionV8(a): e / z for a, e in exps.items()}

    def predict_proba(self, features: Mapping[str, float]) -> Dict[ActionV8, float]:
        raw = self._softmax(features)
        k = len(raw)
        t = min(max(self.uniform_mix, 0.0), 1.0)
        return {a: (1.0 - t) * p + t / k for a, p in raw.items()}

    def update(
        self,
        features: Mapping[str, float],
        action: ActionV8,
        advantage: float,
        lr: float = 0.1,
        entropy_beta: float = 0.01,
    ) -> None:
        """One REINFORCE step with entropy regularization.

        Gradient = advantage · ∇log π(a|s) + entropy_beta · ∇H(π(·|s)).
        The entropy term (∂H/∂z_k = -p_k(log p_k + H)) counters the
        deterministic collapse repeated same-direction updates would cause —
        the prior must keep exploring; the ε-floor in select_action is the
        second, independent safety net. entropy_beta=0 recovers plain
        REINFORCE.
        """
        probs = self._softmax(features)
        entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
        for a in ActionV8.all():
            p_a = probs[a]
            grad = advantage * ((1.0 if a == action else 0.0) - p_a)
            if entropy_beta > 0.0:
                grad += entropy_beta * (-p_a * (math.log(p_a + 1e-12) + entropy))
            self.bias[a.value] += lr * grad
            for f in self.feature_names:
                self.weights[a.value][f] += lr * grad * float(features.get(f, 0.0))

    def update_round(
        self,
        episodes: Sequence[Mapping],
        lr: float = 0.1,
        entropy_beta: float = 0.01,
    ) -> None:
        """按轮更新：episodes 元素含 features/action/advantage。"""
        for e in episodes:
            self.update(e["features"], ActionV8(e["action"]),
                        float(e["advantage"]), lr, entropy_beta)

    def to_dict(self) -> Dict:
        return {"feature_names": list(self.feature_names),
                "weights": self.weights, "bias": self.bias,
                "uniform_mix": self.uniform_mix}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ActionPriorV8":
        p = cls(feature_names=list(d["feature_names"]),
                uniform_mix=float(d.get("uniform_mix", 0.06)))
        p.weights = {a: dict(w) for a, w in d["weights"].items()}
        p.bias = dict(d["bias"])
        return p


# ── 选择（ε-greedy / UCB）───────────────────────────────────────────────────

def select_action(
    values: Mapping[ActionV8, float],
    *,
    epsilon: float = 0.0,
    ucb_c: float = 0.0,
    counts: Optional[Mapping[ActionV8, int]] = None,
    tie_margin: float = 0.0,
    prior_probs: Optional[Mapping[ActionV8, float]] = None,
    rng: Optional[random.Random] = None,
) -> ActionV8:
    """Greedy over Q with ε-exploration / UCB, honest about the noise floor.

    纯贪心不探索（owner 修正案）：epsilon>0 或 ucb_c>0 至少择一用于训练期。
    tie_margin：n=1 估值的噪声底 —— 与最优值差距落在 margin 内的动作视为
    统计打平，交给 prior_probs 裁决而不是假装 lookahead 分辨得出来。
    """
    if not values:
        raise ValueError("select_action requires non-empty values")
    rng = rng or random.Random()
    actions = list(values)
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(actions)
    if ucb_c > 0.0 and counts is not None:
        total = sum(max(int(counts.get(a, 0)), 0) for a in actions) + 1
        adjusted = {
            a: values[a] + ucb_c * math.sqrt(math.log(total) / (int(counts.get(a, 0)) + 1))
            for a in actions
        }
        return max(actions, key=lambda a: (adjusted[a], -actions.index(a)))
    best = max(actions, key=lambda a: (values[a], -actions.index(a)))
    if tie_margin > 0.0 and prior_probs:
        tied = [a for a in actions if values[best] - values[a] <= tie_margin]
        if len(tied) > 1:
            return max(tied, key=lambda a: (float(prior_probs.get(a, 0.0)), -actions.index(a)))
    return best


# ── 预算算术 ────────────────────────────────────────────────────────────────

def lookahead_generation_multiplier(
    decision_points: float,
    n_actions: int = 3,
    samples_per_action: int = 1,
    depth: int = 1,
) -> float:
    """生成调用倍率（相对单次推理）：预注册的同预算对照用它对齐 best-of-N。"""
    if min(decision_points, n_actions, samples_per_action, depth) <= 0:
        raise ValueError("all budget factors must be positive")
    return 1.0 + decision_points * ((n_actions * samples_per_action) ** depth - 1)
