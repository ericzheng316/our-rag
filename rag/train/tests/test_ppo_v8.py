"""ACEC v8 PPO 数学层单测 — CPU 可跑，不触碰 vllm/transformers。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ppo_rsf_vllm_v8 import (
    ValueHead,
    clipped_value_loss,
    explained_variance,
    gae_advantages,
    normalize_advantages,
    ppo_policy_loss,
)


class GAETest(unittest.TestCase):
    def test_matches_independent_recursion(self):
        rewards = [1.0, 0.0, 2.0]
        values = [0.5, 0.4, 0.3]
        for gamma, lam in ((1.0, 1.0), (1.0, 0.95), (0.9, 0.5), (0.0, 0.7)):
            advs, rets = gae_advantages(rewards, values, gamma, lam)
            expected = []
            for t in range(3):
                acc = 0.0
                for k in range(t, 3):
                    next_v = values[k + 1] if k + 1 < 3 else 0.0
                    delta = rewards[k] + gamma * next_v - values[k]
                    acc += (gamma * lam) ** (k - t) * delta
                expected.append(acc)
            for a, e in zip(advs, expected):
                self.assertAlmostEqual(a, e, places=10)
            for r, a, v in zip(rets, advs, values):
                self.assertAlmostEqual(r, a + v, places=10)

    def test_gamma_zero_is_one_step(self):
        advs, _ = gae_advantages([2.0, 3.0], [0.5, 1.0], gamma=0.0, lam=0.9)
        self.assertAlmostEqual(advs[0], 1.5)
        self.assertAlmostEqual(advs[1], 2.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gae_advantages([1.0], [1.0, 2.0])


class PPOLossTest(unittest.TestCase):
    def test_ratio_one_gives_negative_advantage(self):
        lp = torch.tensor([-1.0, -2.0])
        loss = ppo_policy_loss(lp, lp.clone(), torch.tensor(0.7))
        self.assertAlmostEqual(float(loss), -0.7, places=6)

    def test_positive_advantage_clips_upside(self):
        old = torch.tensor([0.0])
        new = old + math.log(2.0)  # ratio = 2
        loss = ppo_policy_loss(new, old, torch.tensor(1.0), clip_eps=0.2)
        self.assertAlmostEqual(float(loss), -1.2, places=6)

    def test_negative_advantage_keeps_penalty_unclipped(self):
        old = torch.tensor([0.0])
        new = old + math.log(2.0)
        loss = ppo_policy_loss(new, old, torch.tensor(-1.0), clip_eps=0.2)
        self.assertAlmostEqual(float(loss), 2.0, places=6)

    def test_value_clip_uses_worse_of_two(self):
        v_old = torch.tensor(0.0)
        ret = torch.tensor(1.0)
        far = torch.tensor(2.0, requires_grad=True)
        loss = clipped_value_loss(far, v_old, ret, clip=0.2)
        # clipped 预测 0.2 → (0.2-1)^2=0.64 vs 原 (2-1)^2=1 → 取更差 0.5*1
        self.assertAlmostEqual(float(loss), 0.5, places=6)
        loss.backward()
        self.assertEqual(float(far.grad), 1.0)


class HeadAndStatsTest(unittest.TestCase):
    def test_value_head_fp32_from_bf16(self):
        vh = ValueHead(16)
        out = vh(torch.randn(16, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, ())

    def test_normalize(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        n = normalize_advantages(x)
        self.assertAlmostEqual(float(n.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(n.std(unbiased=False)), 1.0, places=4)

    def test_explained_variance_perfect_and_constant(self):
        r = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(explained_variance(r, r.clone()), 1.0, places=6)
        self.assertTrue(math.isnan(explained_variance(torch.ones(3), torch.zeros(3))))


if __name__ == "__main__":
    unittest.main()
