import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from belief.acec import ACECBeliefState
from belief.acec.calibration_v4 import (
    ARTIFACT_VERSION,
    LABEL_SCHEMA,
    fit_observation_model_v4,
    load_calibration_artifact_v4,
    save_calibration_artifact_v4,
)
from belief.acec.offline_fit import HitExample


class CalibrationV4Test(unittest.TestCase):
    @staticmethod
    def _builder_module():
        repo_root = Path(__file__).resolve().parents[5]
        builder_path = repo_root / "run_scripts" / "build_acec_calibration_v4.py"
        spec = importlib.util.spec_from_file_location(
            "build_acec_calibration_v4_test", builder_path
        )
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = builder
        spec.loader.exec_module(builder)
        return builder

    @staticmethod
    def _fake_replay_components(selected_contents="Distractor: high NLI", hypothesis="Gold"):
        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                rows = []
                for text in texts:
                    if "gold" in text.casefold():
                        rows.append([1.0, 0.0])
                    else:
                        rows.append([0.0, 1.0])
                return np.asarray(rows, dtype=np.float32)

        class FakeBelief:
            def __init__(self):
                self.coverage_belief = SimpleNamespace(
                    slots=[SimpleNamespace(bound=False, hypothesis=hypothesis)]
                )
                self.labeler = SimpleNamespace(embedder=FakeEmbedder())

            def reset(self, question):
                self.coverage_belief.slots[0].bound = False

            def turn(self, query, new_docs, is_answer=False):
                return SimpleNamespace(
                    slot_scores={0: 0.95},
                    slot_best_docs={0: {"contents": selected_contents}},
                    action=SimpleNamespace(
                        target_slot=0,
                        mode=SimpleNamespace(value="EXPAND"),
                    ),
                )

        class FakeAdapter:
            def gold_titles(self, record):
                return ["Gold"]

        return FakeBelief(), FakeAdapter()

    @staticmethod
    def _record():
        return {
            "id": "example-1",
            "problem": "question",
            "split_querys": [["query"]],
            # The batch contains a gold document, but v4 must label the
            # document selected by max NLI rather than this set membership.
            "docs": [["Gold: lower NLI", "Distractor: high NLI"]],
        }

    def test_batch_gold_does_not_label_selected_distractor_positive(self):
        builder = self._builder_module()
        belief, adapter = self._fake_replay_components()
        rows, _ = builder.replay_record(self._record(), belief, adapter)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].selected_doc_title, "Distractor")
        self.assertEqual(rows[0].assigned_sf_title, "Gold")
        self.assertEqual(rows[0].label_status, "negative")
        self.assertFalse(rows[0].hit_example().is_hit)

    def test_selected_gold_document_is_positive(self):
        builder = self._builder_module()
        belief, adapter = self._fake_replay_components("Gold: selected")
        rows, _ = builder.replay_record(self._record(), belief, adapter)
        self.assertEqual(rows[0].label_status, "positive")
        self.assertTrue(rows[0].hit_example().is_hit)

    def test_unmatched_slot_is_unknown_unless_explicitly_negative(self):
        builder = self._builder_module()
        belief, adapter = self._fake_replay_components(hypothesis="Unrelated")
        rows, _ = builder.replay_record(self._record(), belief, adapter)
        self.assertEqual(rows[0].label_status, "unknown")
        self.assertIsNone(rows[0].hit_example())

        belief, adapter = self._fake_replay_components(hypothesis="Unrelated")
        rows, _ = builder.replay_record(
            self._record(),
            belief,
            adapter,
            unmatched_slot_policy="negative",
        )
        self.assertEqual(rows[0].label_status, "negative")
        self.assertEqual(rows[0].label_reason, "unmatched_slot_assumed_negative")

    def test_live_turn_exposes_the_exact_max_nli_document(self):
        class DummyEmbedder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class DummyNLI:
            def score(self, premise, hypothesis):
                return 0.9 if premise.startswith("Distractor") else 0.2

        belief = ACECBeliefState(DummyEmbedder(), DummyNLI())
        belief.reset("question")
        result = belief.turn(
            query="Gold",
            new_docs=[
                {"contents": "Gold: supporting evidence"},
                {"contents": "Distractor: selected evidence"},
            ],
        )
        self.assertEqual(result.slot_scores[0], 0.9)
        self.assertEqual(result.slot_best_docs[0]["contents"], "Distractor: selected evidence")

    def test_artifact_round_trip_requires_v4_label_schema(self):
        examples = [
            HitExample("EXPAND", "tgt", False, 0.9, True),
            HitExample("EXPAND", "tgt", False, 0.1, False),
            HitExample("EXPAND", "inc", False, 0.8, True),
            HitExample("EXPAND", "inc", False, 0.2, False),
        ] * 6
        model, hit_rates, _ = fit_observation_model_v4(examples)
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            save_calibration_artifact_v4(
                handle.name,
                model,
                hit_rates,
                None,
                {
                    "dataset": "hotpotqa",
                    "recommended_k_mode": "fixed",
                    "fixed_k": 2,
                },
                {"gate": {"pass": True}},
            )
            loaded = load_calibration_artifact_v4(handle.name)
            self.assertEqual(loaded.metadata["fixed_k"], 2)

            handle.seek(0)
            payload = json.load(handle)
            self.assertEqual(payload["artifact_version"], ARTIFACT_VERSION)
            self.assertEqual(payload["label_schema"], LABEL_SCHEMA)
            payload["label_schema"] = "batch_any_gold_title_v1"
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(ValueError):
                load_calibration_artifact_v4(handle.name)


if __name__ == "__main__":
    unittest.main()
