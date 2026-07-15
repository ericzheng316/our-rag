import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from belief.acec.calibration_v3 import (
    ARTIFACT_VERSION,
    MonotonicPlattCalibrator,
    fit_observation_model_v3,
    load_calibration_artifact_v3,
    save_calibration_artifact_v3,
)
from belief.acec.offline_fit import HitExample, KExample


class CalibrationV3Test(unittest.TestCase):
    @staticmethod
    def _builder_module():
        repo_root = Path(__file__).resolve().parents[5]
        builder_path = repo_root / "run_scripts" / "build_acec_calibration_v3.py"
        spec = importlib.util.spec_from_file_location(
            "build_acec_calibration_v3_test", builder_path
        )
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(builder)
        return builder

    def test_platt_calibration_cannot_reverse_nli_order(self):
        calibrator = MonotonicPlattCalibrator.fit(
            [0.02, 0.05, 0.10, 0.20, 0.75, 0.85, 0.95, 0.99],
            [False, False, False, False, True, True, True, True],
        )
        predictions = [calibrator.predict(score) for score in np.linspace(0.01, 0.99, 50)]
        self.assertGreaterEqual(calibrator.slope, 0.0)
        self.assertTrue(all(a <= b for a, b in zip(predictions, predictions[1:])))
        self.assertGreater(predictions[-1], predictions[0])

    def test_target_prior_does_not_leak_into_incidental_slot(self):
        examples = []
        for role in ("tgt", "inc"):
            for score, label in ((0.05, False), (0.10, False), (0.90, True), (0.95, True)):
                examples.extend(
                    [HitExample("EXPAND", role, False, score, label) for _ in range(6)]
                )
        model, _, _ = fit_observation_model_v3(examples)

        target_low_prior = model.hit_posterior(0.8, 0.1, "tgt", False)
        target_high_prior = model.hit_posterior(0.8, 0.9, "tgt", False)
        incidental_low_prior = model.hit_posterior(0.8, 0.1, "inc", False)
        incidental_high_prior = model.hit_posterior(0.8, 0.9, "inc", False)
        self.assertGreater(target_high_prior, target_low_prior)
        self.assertAlmostEqual(incidental_high_prior, incidental_low_prior, places=12)

    def test_unseen_action_uses_pooled_target_rate(self):
        examples = [
            HitExample("EXPAND", "tgt", False, 0.9, True),
            HitExample("EXPAND", "tgt", False, 0.1, False),
            HitExample("DECOMPOSE", "tgt", False, 0.9, True),
            HitExample("DECOMPOSE", "tgt", False, 0.1, False),
            HitExample("DECOMPOSE", "tgt", False, 0.1, False),
            HitExample("DECOMPOSE", "tgt", False, 0.1, False),
        ]
        _, hit_rates, counts = fit_observation_model_v3(examples)
        pooled = (2.0 + 1.0) / (6.0 + 2.0)
        self.assertAlmostEqual(hit_rates["REWRITE"], pooled)
        self.assertEqual(counts["REWRITE"]["fallback"], "pooled_target")
        self.assertNotEqual(hit_rates["REWRITE"], 0.5)

    def test_title_normalization_and_constant_k_auto_mode(self):
        builder = self._builder_module()
        self.assertEqual(builder.normalize_title("  Foo_Bar  "), "foo bar")
        examples = [KExample(np.asarray([float(index), 1.0]), 2) for index in range(5)]
        mode, fixed_k, predictor, payload = builder.select_k_strategy(
            "auto", examples, k_max=4, fixed_k=1, neighbors=3
        )
        self.assertEqual((mode, fixed_k), ("fixed", 2))
        self.assertIsNone(predictor)
        self.assertIsNone(payload)

    def test_artifact_round_trip_and_old_version_rejection(self):
        examples = [
            HitExample("EXPAND", "tgt", False, 0.9, True),
            HitExample("EXPAND", "tgt", False, 0.1, False),
            HitExample("EXPAND", "inc", False, 0.8, True),
            HitExample("EXPAND", "inc", False, 0.2, False),
        ] * 6
        model, hit_rates, _ = fit_observation_model_v3(examples)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            save_calibration_artifact_v3(
                handle.name,
                model,
                hit_rates,
                None,
                {"dataset": "hotpotqa", "recommended_k_mode": "fixed", "fixed_k": 2},
                {"gate": {"pass": True}},
            )
            loaded = load_calibration_artifact_v3(handle.name)
            self.assertEqual(loaded.metadata["fixed_k"], 2)
            self.assertGreaterEqual(
                loaded.observation_model.hit_posterior(0.9, 0.5, "tgt", False),
                loaded.observation_model.hit_posterior(0.1, 0.5, "tgt", False),
            )

            handle.seek(0)
            payload = json.load(handle)
            payload["artifact_version"] = ARTIFACT_VERSION - 1
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(ValueError):
                load_calibration_artifact_v3(handle.name)


if __name__ == "__main__":
    unittest.main()
