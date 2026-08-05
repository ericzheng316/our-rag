"""v8 lookahead 数学核心单测 — 纯 stdlib，毫秒级。"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from belief.acec.lookahead_v8 import (
    ActionPriorV8,
    ActionV8,
    SlotV8,
    action_values,
    expected_return,
    lookahead_generation_multiplier,
    select_action,
)


class SlotPosteriorTest(unittest.TestCase):
    def test_neutral_evidence_is_identity(self):
        s = SlotV8("d", p_bound=0.3)
        s.update(0.5)
        self.assertAlmostEqual(s.p_bound, 0.3, places=9)

    def test_supportive_and_contrary_evidence(self):
        s = SlotV8("d", p_bound=0.3)
        up = s.update(0.9)
        self.assertGreater(up, 0.3)
        down = s.update(0.1)
        self.assertAlmostEqual(down, 0.3, places=9)  # 对称证据相互抵消

    def test_weight_zero_disables(self):
        s = SlotV8("d", p_bound=0.42)
        s.update(0.99, weight=0.0)
        self.assertAlmostEqual(s.p_bound, 0.42, places=9)

    def test_clamped(self):
        s = SlotV8("d", p_bound=0.5)
        for _ in range(200):
            s.update(0.999)
        self.assertLess(s.p_bound, 1.0)


class ExpectationTest(unittest.TestCase):
    def test_single_sample_passthrough(self):
        self.assertEqual(expected_return([0.73]), 0.73)

    def test_mean(self):
        self.assertAlmostEqual(expected_return([0.2, 0.4, 0.9]), 0.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            expected_return([])

    def test_action_values_subtract_costs(self):
        q = action_values(
            {ActionV8.QUERY_TARGET: 0.8, ActionV8.ANSWER: 0.7},
            costs={ActionV8.QUERY_TARGET: 0.05},
        )
        self.assertAlmostEqual(q[ActionV8.QUERY_TARGET], 0.75)
        self.assertAlmostEqual(q[ActionV8.ANSWER], 0.7)


class SelectionTest(unittest.TestCase):
    VALUES = {ActionV8.QUERY_TARGET: 0.2, ActionV8.PIVOT_NEXT: 0.9, ActionV8.ANSWER: 0.5}

    def test_greedy_argmax(self):
        self.assertEqual(select_action(self.VALUES), ActionV8.PIVOT_NEXT)

    def test_epsilon_one_explores(self):
        rng = random.Random(7)
        picks = {select_action(self.VALUES, epsilon=1.0, rng=rng) for _ in range(60)}
        self.assertEqual(picks, set(ActionV8.all()))

    def test_ucb_prefers_unvisited(self):
        counts = {ActionV8.QUERY_TARGET: 0, ActionV8.PIVOT_NEXT: 500, ActionV8.ANSWER: 500}
        pick = select_action(self.VALUES, ucb_c=2.0, counts=counts)
        self.assertEqual(pick, ActionV8.QUERY_TARGET)

    def test_tie_margin_defers_to_prior(self):
        close = {ActionV8.QUERY_TARGET: 0.80, ActionV8.PIVOT_NEXT: 0.81, ActionV8.ANSWER: 0.2}
        prior = {ActionV8.QUERY_TARGET: 0.7, ActionV8.PIVOT_NEXT: 0.2, ActionV8.ANSWER: 0.1}
        # 差 0.01 落在噪声底内 → 先验裁决
        self.assertEqual(
            select_action(close, tie_margin=0.03, prior_probs=prior),
            ActionV8.QUERY_TARGET)
        # 差距拉大到 margin 之外 → 贪心不受先验影响
        far = {ActionV8.QUERY_TARGET: 0.5, ActionV8.PIVOT_NEXT: 0.81, ActionV8.ANSWER: 0.2}
        self.assertEqual(
            select_action(far, tie_margin=0.03, prior_probs=prior),
            ActionV8.PIVOT_NEXT)


class ActionPriorTest(unittest.TestCase):
    def test_uniform_at_init(self):
        p = ActionPriorV8(feature_names=["cov"])
        probs = p.predict_proba({"cov": 0.4})
        for v in probs.values():
            self.assertAlmostEqual(v, 1 / 3, places=9)

    def test_learns_rewarded_action(self):
        # 可学性测试（非训练制度）：lr=0.2 x 50 步故意打满信号验证梯度方向
        p = ActionPriorV8(feature_names=["cov"])
        for _ in range(50):
            p.update({"cov": 1.0}, ActionV8.ANSWER, advantage=1.0, lr=0.2, entropy_beta=0.0)
        probs = p.predict_proba({"cov": 1.0})
        self.assertGreater(probs[ActionV8.ANSWER], 0.8)

    def test_uniform_mix_floors_collapse(self):
        # 激进同向更新下，混合地板保证探索质量（熵正则只延缓内部塌缩）
        p = ActionPriorV8(feature_names=["cov"])
        for _ in range(300):
            p.update({"cov": 1.0}, ActionV8.ANSWER, advantage=1.0, lr=0.2, entropy_beta=0.05)
        probs = p.predict_proba({"cov": 1.0})
        self.assertEqual(max(probs, key=probs.get), ActionV8.ANSWER)  # 方向仍正确
        # 硬地板：min ≥ uniform_mix/K = 0.06/3 = 0.02
        self.assertGreaterEqual(min(probs.values()), 0.06 / 3 - 1e-9)
        # 内部 softmax 确实塌了（说明地板来自混合，不是熵正则撑住的）
        self.assertLess(min(p._softmax({"cov": 1.0}).values()), 0.01)

    def test_round_update_and_roundtrip(self):
        p = ActionPriorV8(feature_names=["a", "b"])
        p.update_round(
            [{"features": {"a": 1.0, "b": 0.0}, "action": "pivot_next", "advantage": 2.0}] * 30,
            lr=0.1,
        )
        probs = p.predict_proba({"a": 1.0, "b": 0.0})
        self.assertEqual(max(probs, key=probs.get), ActionV8.PIVOT_NEXT)
        clone = ActionPriorV8.from_dict(p.to_dict())
        probs2 = clone.predict_proba({"a": 1.0, "b": 0.0})
        for a in ActionV8.all():
            self.assertAlmostEqual(probs[a], probs2[a], places=12)


class BudgetTest(unittest.TestCase):
    def test_hotpot_depth1(self):
        # 1.74 决策点 × 3 动作 × n=1、深度 1 → 1 + 1.74×2 = 4.48×
        self.assertAlmostEqual(
            lookahead_generation_multiplier(1.74, 3, 1, 1), 4.48, places=6)

    def test_depth_scaling_and_validation(self):
        self.assertAlmostEqual(lookahead_generation_multiplier(1.0, 3, 1, 2), 9.0)
        with self.assertRaises(ValueError):
            lookahead_generation_multiplier(0, 3, 1, 1)


if __name__ == "__main__":
    unittest.main()
