import unittest

from grpo_estimator_v7 import (
    CoverageRewardModeV7,
    grpo_loss_v7,
    terminal_potential_correction_v7,
)


class GRPOEstimatorV7ContractTest(unittest.TestCase):
    def test_strict_pbrs_zeros_terminal_potential(self):
        self.assertAlmostEqual(
            terminal_potential_correction_v7(
                0.8, 0.3, CoverageRewardModeV7.STRICT_PBRS
            ),
            -0.24,
        )
        self.assertEqual(
            terminal_potential_correction_v7(
                0.8, 0.3, CoverageRewardModeV7.LEGACY_COVERAGE_AUX
            ),
            0.0,
        )

    def test_predicted_gain_cannot_be_paid_twice(self):
        with self.assertRaisesRegex(ValueError, "forbids"):
            grpo_loss_v7(None, predicted_gain_bonus=0.1)


if __name__ == "__main__":
    unittest.main()
