import math
import unittest

import torch

from grpo_estimator_v2 import (
    RewardConfig,
    clipped_grpo_token_loss,
    completion_token_logprobs,
)


class GrpoEstimatorV2Test(unittest.TestCase):
    def test_first_update_policy_gradient_matches_unclipped_objective(self):
        advantage = torch.tensor(1.7)

        policy_v2 = torch.tensor([-0.4, -1.2, -2.0], requires_grad=True)
        loss_v2, _ = clipped_grpo_token_loss(
            policy_v2,
            policy_v2.detach().clone(),
            policy_v2.detach().clone(),
            advantage,
            kl_coef=0.0,
        )
        loss_v2.backward()

        policy_legacy = policy_v2.detach().clone().requires_grad_(True)
        loss_legacy = -advantage * policy_legacy.mean()
        loss_legacy.backward()

        self.assertTrue(torch.allclose(policy_v2.grad, policy_legacy.grad))

    def test_kl_is_zero_for_identical_policy_and_reference(self):
        policy = torch.tensor([-1.0, -2.0], requires_grad=True)
        old = policy.detach().clone()
        loss, stats = clipped_grpo_token_loss(
            policy, old, policy.detach().clone(), torch.tensor(1.0)
        )
        self.assertAlmostEqual(float(stats["kl"]), 0.0, places=7)
        self.assertAlmostEqual(float(stats["ratio_mean"]), 1.0, places=7)
        loss.backward()
        self.assertIsNotNone(policy.grad)

    def test_positive_advantage_ratio_is_clipped(self):
        policy = torch.tensor([math.log(2.0)], requires_grad=True)
        old = torch.tensor([0.0])
        loss, stats = clipped_grpo_token_loss(
            policy,
            old,
            policy.detach().clone(),
            torch.tensor(1.0),
            clip_eps=0.2,
            kl_coef=0.0,
        )
        self.assertAlmostEqual(float(loss), -1.2, places=6)
        self.assertAlmostEqual(float(stats["clip_fraction"]), 1.0, places=6)

    def test_kl_estimator_is_pointwise_nonnegative(self):
        policy = torch.tensor([-0.2, -1.4], requires_grad=True)
        reference = torch.tensor([-1.0, -0.3])
        _, stats = clipped_grpo_token_loss(
            policy,
            policy.detach().clone(),
            reference,
            torch.tensor(0.0),
        )
        self.assertGreaterEqual(float(stats["kl"]), 0.0)

    def test_completion_logprob_alignment(self):
        logits = torch.zeros((1, 4, 3))
        logits[0, 1, 2] = 4.0  # predicts first output token
        logits[0, 2, 1] = 3.0  # predicts second output token
        output_ids = torch.tensor([2, 1])
        actual = completion_token_logprobs(logits, output_ids, temperature=1.0)
        expected0 = torch.log_softmax(logits[0, 1], dim=-1)[2]
        expected1 = torch.log_softmax(logits[0, 2], dim=-1)[1]
        self.assertTrue(torch.allclose(actual, torch.stack([expected0, expected1]).float()))

    def test_reward_config_is_identical_across_coverage_sources(self):
        config = RewardConfig(coverage=0.3, retrieval_cost=0.05)
        gold_delta = 0.5
        acec_delta = 0.5
        self.assertEqual(
            config.retrieval_reward(gold_delta),
            config.retrieval_reward(acec_delta),
        )


if __name__ == "__main__":
    unittest.main()
