import unittest

import numpy as np

from belief.acec.action_labeler import ActionLabel, ActionMode
from belief.acec.calibration_v2 import FixedKPredictor
from belief.acec.config import ACECConfig
from belief.acec.runtime_v5 import (
    ACECBeliefStateV5,
    CoverageBeliefV5,
    RuntimeDiagnosticsV5,
)


class RecordingObservationModel:
    def __init__(self):
        self.calls = []

    def gain_posterior(self, support, novelty, pi_a, role, bound):
        self.calls.append((support, novelty, pi_a, role, bound))
        return min(max(support * novelty, 0.0), 1.0)


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeNLIScorer:
    def score(self, premise, hypothesis):
        return 0.9


class RuntimeV5Test(unittest.TestCase):
    def test_coverage_update_requires_runtime_novelty(self):
        config = ACECConfig(k_max=1)
        model = RecordingObservationModel()
        belief = CoverageBeliefV5(config, FixedKPredictor(1, 1), model)
        belief.reset("question")
        slot = belief.spawn_slot("first requirement")
        action = ActionLabel(ActionMode.DECOMPOSE, slot)
        belief.step(action, {slot: 0.8}, {slot: 0.5})
        self.assertAlmostEqual(belief.p[slot], 0.4)
        self.assertEqual(model.calls[0][1], 0.5)
        with self.assertRaises(ValueError):
            belief.step(action, {slot: 0.8}, {})

    def test_state_matches_offline_selected_document_novelty_history(self):
        config = ACECConfig(k_max=1, tau_new=0.1, bound_threshold=1.0)
        model = RecordingObservationModel()
        diagnostics = RuntimeDiagnosticsV5()
        belief = ACECBeliefStateV5(
            FakeEmbedder(),
            FakeNLIScorer(),
            model,
            config=config,
            k_predictor=FixedKPredictor(1, 1),
            diagnostics=diagnostics,
        )
        belief.reset("Where was Alice born?")
        document = {"id": "alice", "contents": "Alice: Alice was born in Paris."}
        first = belief.turn("Alice birthplace", [document])
        second = belief.turn("Alice birthplace", [document])
        self.assertGreater(first.delta_coverage, 0.0)
        self.assertAlmostEqual(second.delta_coverage, 0.0)
        self.assertEqual(model.calls[0][1], 1.0)
        self.assertEqual(model.calls[1][1], 0.0)
        summary = diagnostics.summary()
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["observations"], 2)
        self.assertEqual(summary["exact_repeat_fraction"], 0.5)

    def test_same_turn_slots_do_not_mark_each_other_as_prior_documents(self):
        config = ACECConfig(k_max=2, tau_new=0.99, bound_threshold=1.0)
        model = RecordingObservationModel()
        belief = ACECBeliefStateV5(
            FakeEmbedder(),
            FakeNLIScorer(),
            model,
            config=config,
            k_predictor=FixedKPredictor(2, 2),
            diagnostics=RuntimeDiagnosticsV5(),
        )
        belief.reset("question")
        first_document = {"id": "a", "contents": "Alice was born in Paris."}
        second_document = {"id": "b", "contents": "Bob was born in Rome."}
        belief.turn("first requirement", [first_document])
        # Force a second slot while retaining the first; both slots select the
        # same new document on this turn and must see novelty relative only to
        # earlier turns, not relative to one another.
        belief.labeler.tau_new = 1.1
        calls_before = len(model.calls)
        belief.turn("second requirement", [second_document])
        same_turn_calls = model.calls[calls_before:]
        self.assertEqual(len(same_turn_calls), 2)
        self.assertAlmostEqual(same_turn_calls[0][1], same_turn_calls[1][1])


if __name__ == "__main__":
    unittest.main()
