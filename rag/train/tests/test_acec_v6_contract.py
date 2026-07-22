import unittest

from acec_v6_contract import (
    AdaptiveKLController,
    EvidenceAnswerGuard,
    effective_kl_coefficient,
)


class EvidenceAnswerGuardTest(unittest.TestCase):
    def test_correct_answer_is_never_penalized(self):
        guard = EvidenceAnswerGuard(min_coverage=0.5, wrong_answer_penalty=0.1)
        self.assertEqual(guard.reward_adjustment(True, 0.0), 0.0)
        self.assertEqual(guard.reward_adjustment(True, 1.0), 0.0)

    def test_only_low_coverage_wrong_answer_is_penalized(self):
        guard = EvidenceAnswerGuard(min_coverage=0.5, wrong_answer_penalty=0.1)
        self.assertEqual(guard.reward_adjustment(False, 0.49), -0.1)
        self.assertEqual(guard.reward_adjustment(False, 0.5), 0.0)


class AdaptiveKLControllerTest(unittest.TestCase):
    def test_never_drops_below_v5_base(self):
        controller = AdaptiveKLController(base_coef=0.01)
        for _ in range(10):
            controller.update(0.0)
        self.assertEqual(controller.coef, 0.01)

    def test_high_kl_increases_then_recovers(self):
        controller = AdaptiveKLController(base_coef=0.01)
        self.assertAlmostEqual(controller.update(0.09), 0.0125)
        recovered = controller.update(0.01)
        self.assertGreaterEqual(recovered, 0.01)
        self.assertLess(recovered, 0.0125)

    def test_coefficient_is_capped(self):
        controller = AdaptiveKLController(base_coef=0.01, max_coef=0.02)
        for _ in range(20):
            controller.update(1.0)
        self.assertEqual(controller.coef, 0.02)


class EffectiveKLCoefficientTest(unittest.TestCase):
    def test_answer_turn_is_anchored_more_strongly(self):
        self.assertEqual(
            effective_kl_coefficient(
                0.01, turn_kind="answer", answer_multiplier=2.0
            ),
            0.02,
        )
        self.assertEqual(
            effective_kl_coefficient(
                0.01, turn_kind="retrieval", answer_multiplier=2.0
            ),
            0.01,
        )

    def test_invalid_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "answer_multiplier"):
            effective_kl_coefficient(
                0.01, turn_kind="answer", answer_multiplier=0.5
            )


if __name__ == "__main__":
    unittest.main()
