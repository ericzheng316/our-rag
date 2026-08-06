"""ACEC v8 PPO 数学层单测 — CPU 可跑，不触碰 vllm/transformers。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ppo_rsf_vllm_v8 import (
    MLPValueHead,
    ValueHead,
    _turn_forward,
    _turns_forward_batched,
    make_value_head,
    ppo_update,
    value_feat_dim,
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

    def test_value_head_zero_init(self):
        vh = ValueHead(64)
        out = vh(torch.randn(64, dtype=torch.bfloat16) * 50)
        self.assertEqual(float(out), 0.0)

    def test_mlp_head_zero_init_and_grad(self):
        vh = MLPValueHead(64, 32)
        h = torch.randn(64, dtype=torch.bfloat16) * 50
        out = vh(h)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(float(out), 0.0)          # 末层零初始化 → V≡0 起步
        loss = (vh(h) - 1.0) ** 2
        loss.backward()
        g = vh.net[-1].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(float(g.abs().sum()), 0.0)  # 梯度经末层正常流动

    def test_factory_switches(self):
        self.assertIsInstance(make_value_head(16, 0), ValueHead)
        self.assertIsInstance(make_value_head(16, 8), MLPValueHead)

    def test_normalize(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        n = normalize_advantages(x)
        self.assertAlmostEqual(float(n.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(n.std(unbiased=False)), 1.0, places=4)

    def test_explained_variance_perfect_and_constant(self):
        r = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(explained_variance(r, r.clone()), 1.0, places=6)
        self.assertTrue(math.isnan(explained_variance(torch.ones(3), torch.zeros(3))))


class _FakeBackbone(torch.nn.Module):
    class _Out:
        def __init__(self, h):
            self.last_hidden_state = h

    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None):
        return self._Out(self.emb(input_ids))


class _FakeCausalLM(torch.nn.Module):
    """HF CausalLM 形状的确定性假模型：.model(backbone) + .lm_head。

    逐位置独立（严格因果、pad 不敏感），所以批版（backbone + completion 行
    lm_head 切片）与逐条完整调用的任何数值差异只能来自 padding / gather /
    索引算术 —— 正是这组测试要守的东西。完整 forward 与子模块共享参数，
    等价断言因此同时验证了 lm_head 切片路径。
    """

    class _Out:
        def __init__(self, logits, hidden):
            self.logits = logits
            self.hidden_states = [hidden]

    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        torch.manual_seed(7)
        self.model = _FakeBackbone(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.model(input_ids).last_hidden_state
        return self._Out(self.lm_head(h), h)


class MicroBatchEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeCausalLM()
        self.vhead = torch.nn.Linear(16, 1)
        torch.nn.init.normal_(self.vhead.weight)

        def vh(x):
            return self.vhead(x).squeeze(-1)
        self.vh = vh
        g = torch.Generator().manual_seed(3)
        self.pairs = [
            (torch.randint(0, 64, (li,), generator=g),
             torch.randint(0, 64, (lo,), generator=g))
            for li, lo in ((5, 3), (9, 1), (2, 6))
        ]

    def test_batched_matches_single(self):
        lps_b, vs_b, _ = _turns_forward_batched(
            self.model, self.vh, self.pairs, need_grad=False)
        for (inp, out), lp_b, v_b in zip(self.pairs, lps_b, vs_b):
            lp_s, v_s = _turn_forward(self.model, self.vh, inp, out, need_grad=False)
            self.assertTrue(torch.allclose(lp_b, lp_s, atol=1e-5),
                            f"logprob 批/单不一致: {lp_b} vs {lp_s}")
            self.assertTrue(torch.allclose(v_b, v_s, atol=1e-5))

    def test_entropy_positive_and_shapes(self):
        lps, vs, ents = _turns_forward_batched(
            self.model, self.vh, self.pairs, need_grad=False, want_entropy=True)
        for (_, out), lp, e in zip(self.pairs, lps, ents):
            self.assertEqual(lp.numel(), out.numel())
            self.assertGreater(float(e), 0.0)

    def test_grad_flows_through_batch(self):
        lps, vs, _ = _turns_forward_batched(
            self.model, self.vh, self.pairs, need_grad=True)
        loss = sum(lp.mean() for lp in lps) + sum(v for v in vs)
        loss.backward()
        self.assertIsNotNone(self.model.lm_head.weight.grad)
        self.assertGreater(float(self.model.lm_head.weight.grad.abs().sum()), 0.0)


class GroupBaselineModeTest(unittest.TestCase):
    """组基线 / 影子 / 特征收集的端到端行为（fake 模型）。"""

    def _setup(self, feat_dim=16):
        model = _FakeCausalLM()
        for p_ in model.parameters():
            p_.requires_grad_(True)
        vhead = MLPValueHead(feat_dim, 8)
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": 1e-3},
             {"params": vhead.parameters(), "lr": 1e-3}])
        g = torch.Generator().manual_seed(4)

        def turn(li, lo, r):
            return (torch.randint(0, 64, (li,), generator=g),
                    torch.randint(0, 64, (lo,), generator=g), r, None, 0.7)
        # 2 题 × 2 rollouts：组内有对比（1.0 vs 0.0）
        trajs = [[turn(6, 3, -0.005), turn(8, 2, 1.0)],
                 [turn(6, 3, -0.005), turn(7, 2, 0.0)],
                 [turn(5, 2, 1.0)],
                 [turn(5, 3, 0.0)]]
        gids = ["qA", "qA", "qB", "qB"]
        return model, vhead, opt, [[t] for t in trajs], gids

    def _kw(self):
        return dict(gamma=0.99, gae_lambda=0.95, clip_eps=0.2, value_clip=0.2,
                    value_coef=0.5, kl_coef=0.0, entropy_coef=0.0, ppo_epochs=1,
                    max_grad_norm=1.0, max_engine_logprob_mae=None,
                    micro_turns=1, scan_batch=4)

    def test_group_baseline_with_collect(self):
        model, vhead, opt, trajs, gids = self._setup(feat_dim=32)  # pool_last_ln = 2*hidden
        stats = ppo_update(model, vhead, opt, trajs, group_ids=gids,
                           group_baseline=True, value_variant="pool_last_ln",
                           collect_value_samples=True, **self._kw())
        self.assertEqual(stats["n_turns"], 6)
        feats, G = stats["value_samples"]
        self.assertEqual(tuple(feats.shape), (6, 32))  # 2*hidden=32
        self.assertEqual(G.numel(), 6)
        self.assertTrue(any(abs(float(x) - 1.0) < 0.05 for x in G))

    def test_shadow_freezes_vhead_in_main_loss(self):
        model, vhead, opt, trajs, gids = self._setup()
        before = {k: v.clone() for k, v in vhead.state_dict().items()}
        ppo_update(model, vhead, opt, trajs, group_ids=gids,
                   group_baseline=True, train_value=False,
                   value_variant="last", **self._kw())
        for k, v in vhead.state_dict().items():
            self.assertTrue(torch.equal(before[k], v),
                            f"影子模式下 vhead.{k} 不应被主循环更新")

    def test_feat_dim_helper(self):
        self.assertEqual(value_feat_dim(4096, "last"), 4096)
        self.assertEqual(value_feat_dim(4096, "pool_last"), 8192)
        self.assertEqual(value_feat_dim(4096, "pool_last_ln"), 8192)


if __name__ == "__main__":
    unittest.main()
