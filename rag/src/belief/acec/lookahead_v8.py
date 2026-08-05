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
    """E[R(s')] from executed-branch judge scores.

    Under greedy decoding and deterministic retrieval one sample per action
    IS the expectation (n=1 exact); with sampling, pass n>=2 scores.
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

    def predict_proba(self, features: Mapping[str, float]) -> Dict[ActionV8, float]:
        logits = self._logits(features)
        m = max(logits.values())
        exps = {a: math.exp(v - m) for a, v in logits.items()}
        z = sum(exps.values())
        return {ActionV8(a): e / z for a, e in exps.items()}

    def update(
        self,
        features: Mapping[str, float],
        action: ActionV8,
        advantage: float,
        lr: float = 0.1,
    ) -> None:
        """∇ log π(a|s) · advantage — one REINFORCE step."""
        probs = self.predict_proba(features)
        for a in ActionV8.all():
            grad = (1.0 if a == action else 0.0) - probs[a]
            self.bias[a.value] += lr * advantage * grad
            for f in self.feature_names:
                self.weights[a.value][f] += lr * advantage * grad * float(features.get(f, 0.0))

    def update_round(
        self,
        episodes: Sequence[Mapping],
        lr: float = 0.1,
    ) -> None:
        """按轮更新：episodes 元素含 features/action/advantage。"""
        for e in episodes:
            self.update(e["features"], ActionV8(e["action"]), float(e["advantage"]), lr)

    def to_dict(self) -> Dict:
        return {"feature_names": list(self.feature_names),
                "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ActionPriorV8":
        p = cls(feature_names=list(d["feature_names"]))
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
    rng: Optional[random.Random] = None,
) -> ActionV8:
    """Greedy over Q with optional ε-exploration or a UCB bonus.

    纯贪心不探索（owner 修正案）：epsilon>0 或 ucb_c>0 至少择一用于训练期。
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
    return max(actions, key=lambda a: (values[a], -actions.index(a)))


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
