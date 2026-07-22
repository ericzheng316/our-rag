import unittest

from outcome_only_contract import (
    OUTCOME_ONLY_REWARD,
    OutcomeOnlyReward,
    validate_outcome_only_reward,
)


class OutcomeOnlyContractTest(unittest.TestCase):
    def test_only_terminal_correctness_can_change_reward(self):
        self.assertEqual(OUTCOME_ONLY_REWARD.answer_reward(True), 1.0)
        self.assertEqual(OUTCOME_ONLY_REWARD.answer_reward(False), 0.0)
        self.assertEqual(OUTCOME_ONLY_REWARD.retrieval_reward(0.0), 0.0)
        self.assertEqual(OUTCOME_ONLY_REWARD.retrieval_reward(1.0), 0.0)
        self.assertEqual(OUTCOME_ONLY_REWARD.format_error, 0.0)

    def test_contract_rejects_every_process_signal(self):
        for config in (
            OutcomeOnlyReward(1.0, 0.3, 0.0, 0.0),
            OutcomeOnlyReward(1.0, 0.0, 0.1, 0.0),
            OutcomeOnlyReward(1.0, 0.0, 0.0, 0.05),
        ):
            with self.assertRaisesRegex(ValueError, "outcome-only requires"):
                validate_outcome_only_reward(config)


if __name__ == "__main__":
    unittest.main()
