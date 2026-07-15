import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from belief.acec.calibration_v2 import (
    ARTIFACT_VERSION,
    FixedKPredictor,
    build_k_predictor,
    fit_observation_model_v2,
    load_calibration_artifact_v2,
    save_calibration_artifact_v2,
)
from belief.acec.offline_fit import HitExample


class CalibrationV2Test(unittest.TestCase):
    def test_replay_uses_binding_state_from_the_scored_turn(self):
        repo_root = Path(__file__).resolve().parents[5]
        builder_path = repo_root / "run_scripts" / "build_acec_calibration_v2.py"
        spec = importlib.util.spec_from_file_location("build_acec_calibration_v2_test", builder_path)
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(builder)

        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeBelief:
            def __init__(self):
                self.coverage_belief = SimpleNamespace(
                    slots=[SimpleNamespace(bound=False, hypothesis="Gold")]
                )
                self.labeler = SimpleNamespace(embedder=FakeEmbedder())
                self.turn_index = 0

            def reset(self, question):
                self.coverage_belief.slots[0].bound = False
                self.turn_index = 0

            def turn(self, query, new_docs, is_answer=False):
                result = SimpleNamespace(
                    slot_scores={0: 0.9},
                    action=SimpleNamespace(
                        target_slot=0,
                        mode=SimpleNamespace(value="EXPAND"),
                    ),
                )
                if self.turn_index == 0:
                    # Mirrors live ACECBeliefState: binding happens after this
                    # turn's observation-model score has already been used.
                    self.coverage_belief.slots[0].bound = True
                self.turn_index += 1
                return result

        class FakeAdapter:
            def gold_titles(self, record):
                return ["Gold"]

        record = {
            "problem": "question",
            "split_querys": [["q1"], ["q2"]],
            "docs": [["Gold: first"], ["Gold: second"]],
        }
        examples, _ = builder.replay_record(record, FakeBelief(), FakeAdapter())
        self.assertEqual([example.bound for example in examples], [False, True])

    def test_hit_rate_uses_target_rows_only(self):
        examples = [
            HitExample("REWRITE", "tgt", False, 0.9, True),
            HitExample("REWRITE", "tgt", False, 0.1, False),
            HitExample("REWRITE", "inc", False, 0.1, False),
            HitExample("REWRITE", "inc", False, 0.1, False),
        ]
        _, hit_rates, counts = fit_observation_model_v2(examples)
        # Beta(1,1) posterior mean over the two target rows: (1+1)/(2+2)=0.5.
        self.assertAlmostEqual(hit_rates["REWRITE"], 0.5)
        self.assertEqual(counts["REWRITE"]["target_examples"], 2)

    def test_fixed_k_predictor(self):
        predictor = FixedKPredictor(k_max=4, fixed_k=2)
        self.assertEqual(predictor.predict("anything"), [0.0, 1.0, 0.0, 0.0])

    def test_predictor_mode_requires_v2_payload(self):
        with self.assertRaises(ValueError):
            build_k_predictor("predictor", 4, embedder=None, artifact=None)

    def test_artifact_round_trip_and_version_rejection(self):
        model, hit_rates, _ = fit_observation_model_v2(
            [
                HitExample("EXPAND", "tgt", False, 0.9, True),
                HitExample("EXPAND", "tgt", False, 0.1, False),
            ]
        )
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            save_calibration_artifact_v2(
                handle.name,
                model,
                hit_rates,
                None,
                {"dataset": "hotpotqa"},
                {"test": {"posterior_auc": 1.0}},
            )
            loaded = load_calibration_artifact_v2(handle.name)
            self.assertEqual(loaded.metadata["dataset"], "hotpotqa")

            handle.seek(0)
            payload = json.load(handle)
            payload["artifact_version"] = ARTIFACT_VERSION + 1
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(ValueError):
                load_calibration_artifact_v2(handle.name)


if __name__ == "__main__":
    unittest.main()
