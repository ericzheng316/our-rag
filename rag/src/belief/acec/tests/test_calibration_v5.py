import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from belief.acec.calibration_v5 import (
    ARTIFACT_VERSION,
    MarginalUtilityExample,
    MonotonicMarginalCalibrator,
    choose_binding_threshold_v5,
    fit_observation_model_v5,
    load_calibration_artifact_v5,
    posterior_quality_metrics_v5,
    save_calibration_artifact_v5,
)
from belief.acec.evidence_standard_v5 import (
    AnnotationScope,
    EvidenceRequirement,
    EvidenceSpecification,
    EvidenceUnit,
    RequirementCoverageHistory,
    assess_selected_document,
    document_novelty_score,
    standard_payload,
)
from belief.acec.k_strategy_v5 import KStrategyV5, select_k_strategy_v5
from belief.acec.offline_fit import KExample


class CalibrationV5Test(unittest.TestCase):
    @staticmethod
    def _requirement():
        return EvidenceRequirement(
            requirement_id="hop:0",
            description="Alice's birthplace",
            evidence_units=(
                EvidenceUnit(
                    unit_id="hop:0:sent:1",
                    text="Alice was born in Paris in 1980.",
                    source_id="Alice",
                ),
            ),
        )

    @staticmethod
    def _builder_module():
        repo_root = Path(__file__).resolve().parents[5]
        builder_path = repo_root / "run_scripts" / "build_acec_calibration_v5.py"
        spec = importlib.util.spec_from_file_location(
            "build_acec_calibration_v5_test", builder_path
        )
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = builder
        spec.loader.exec_module(builder)
        return builder

    def test_standard_targets_marginal_utility_not_source_id(self):
        payload = standard_payload()
        self.assertIn("marginal utility", payload["target"])
        self.assertFalse(payload["source_ids_are_targets"])
        self.assertTrue(payload["static_support_is_intermediate_only"])

    def test_exact_span_support_is_independent_of_title(self):
        assessment = assess_selected_document(
            self._requirement(),
            "Completely Different Source: Alice was born in Paris in 1980.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        self.assertTrue(assessment.assessable)
        self.assertEqual(assessment.coverage_value, 1.0)

        same_title_without_evidence = assess_selected_document(
            self._requirement(),
            "Alice: She became a painter and exhibited in London.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        self.assertTrue(same_title_without_evidence.assessable)
        self.assertEqual(same_title_without_evidence.coverage_value, 0.0)

    def test_open_world_absence_abstains(self):
        assessment = assess_selected_document(
            self._requirement(),
            "Unrelated material.",
            AnnotationScope.SUPPORT_ONLY,
        )
        self.assertFalse(assessment.assessable)
        self.assertIsNone(assessment.coverage_value)

    def test_all_rule_does_not_mark_partial_requirement_as_covered(self):
        requirement = EvidenceRequirement(
            requirement_id="hop:0",
            description="two facts from one source",
            evidence_units=(
                EvidenceUnit("u1", "Alice was born in Paris."),
                EvidenceUnit("u2", "Alice moved to Rome in 1999."),
            ),
            satisfaction_rule="all",
        )
        assessment = assess_selected_document(
            requirement,
            "Alice: Alice was born in Paris. She later became a painter.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        self.assertFalse(assessment.assessable)
        self.assertIsNone(assessment.coverage_value)

    def test_first_coverage_has_gain_and_repeat_has_zero_gain(self):
        requirement = self._requirement()
        static = assess_selected_document(
            requirement,
            "Alice: Alice was born in Paris in 1980.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        history = RequirementCoverageHistory(requirement_count=2)
        first = history.observe(static, "Alice: Alice was born in Paris in 1980.")
        repeat = history.observe(static, "Alice: Alice was born in Paris in 1980.")
        self.assertEqual(first.marginal_utility, 0.5)
        self.assertTrue(first.new_requirement_coverage)
        self.assertEqual(repeat.marginal_utility, 0.0)
        self.assertFalse(repeat.new_requirement_coverage)
        self.assertTrue(repeat.repeated_document)

    def test_same_turn_assessments_share_pre_turn_utility(self):
        first_requirement = self._requirement()
        second_requirement = EvidenceRequirement(
            requirement_id="hop:1",
            description="Bob's birthplace",
            evidence_units=(EvidenceUnit("hop:1:sent:1", "Bob was born in Rome."),),
        )
        first_static = assess_selected_document(
            first_requirement,
            "Alice was born in Paris in 1980.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        second_static = assess_selected_document(
            second_requirement,
            "Bob was born in Rome.",
            AnnotationScope.CLOSED_CANDIDATE_POOL,
        )
        history = RequirementCoverageHistory(requirement_count=2)
        snapshot = set(history.covered_requirement_ids)
        first = history.observe(
            first_static,
            "Alice was born in Paris in 1980.",
            covered_requirement_ids_before=snapshot,
            commit_coverage=False,
        )
        second = history.observe(
            second_static,
            "Bob was born in Rome.",
            covered_requirement_ids_before=snapshot,
            commit_coverage=False,
        )
        self.assertEqual(first.utility_before, 0.0)
        self.assertEqual(second.utility_before, 0.0)
        self.assertEqual(first.marginal_utility, 0.5)
        self.assertEqual(second.marginal_utility, 0.5)

    def test_document_novelty_is_zero_for_exact_repeat(self):
        document = "Alice was born in Paris."
        self.assertEqual(document_novelty_score(document, []), 1.0)
        self.assertAlmostEqual(document_novelty_score(document, [document]), 0.0)
        self.assertGreater(
            document_novelty_score("Bob studied chemistry in Rome.", [document]),
            0.5,
        )

    def test_builder_replay_makes_second_identical_support_zero_gain(self):
        builder = self._builder_module()
        requirement = self._requirement()

        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        class FakeAdapter:
            name = "fake"

            def evidence_specification(self, record):
                return EvidenceSpecification(
                    question_id="q1",
                    requirements=(requirement,),
                    annotation_scope=AnnotationScope.CLOSED_CANDIDATE_POOL,
                    adapter_name=self.name,
                    annotation_version="1",
                )

        class FakeBelief:
            def __init__(self):
                self.labeler = SimpleNamespace(embedder=FakeEmbedder())
                self.coverage_belief = SimpleNamespace(slots=[])

            def reset(self, question):
                self.coverage_belief.slots = []

            def turn(self, query, new_docs, is_answer=False):
                if not self.coverage_belief.slots:
                    self.coverage_belief.slots.append(
                        SimpleNamespace(hypothesis="Alice's birthplace", bound=False)
                    )
                return SimpleNamespace(
                    slot_scores={0: 0.95},
                    slot_best_docs={
                        0: {"contents": "Alice: Alice was born in Paris in 1980."}
                    },
                    action=SimpleNamespace(
                        target_slot=0,
                        mode=SimpleNamespace(value="DECOMPOSE"),
                    ),
                )

        record = {
            "id": "q1",
            "problem": "Where was Alice born?",
            "split_querys": [["Alice birthplace"], ["Alice birthplace"]],
            "docs": [["doc"], ["doc"]],
        }
        rows, k_example = builder.replay_record(record, FakeBelief(), FakeAdapter())
        self.assertEqual(k_example.k_true, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].new_requirement_coverage)
        self.assertGreater(rows[0].marginal_utility, 0.0)
        self.assertFalse(rows[1].new_requirement_coverage)
        self.assertEqual(rows[1].marginal_utility, 0.0)
        self.assertEqual(rows[1].novelty_score, 0.0)
        self.assertTrue(rows[1].repeated_document)

    def test_model_and_artifact_round_trip(self):
        examples = [
            MarginalUtilityExample("EXPAND", "tgt", False, 0.95, 1.0, True, 0.5),
            MarginalUtilityExample("EXPAND", "tgt", True, 0.95, 0.0, False, 0.0),
            MarginalUtilityExample("REWRITE", "tgt", False, 0.10, 1.0, False, 0.0),
            MarginalUtilityExample("DECOMPOSE", "inc", False, 0.90, 0.9, True, 0.5),
        ] * 8
        model, gain_rates, _ = fit_observation_model_v5(examples)
        global_calibrator = model.calibrators["global"]
        self.assertGreaterEqual(global_calibrator.support_slope, 0.0)
        self.assertGreaterEqual(global_calibrator.novelty_slope, 0.0)
        strategy = KStrategyV5(
            mode="fixed",
            k_max=4,
            fixed_k=2,
            predictor_payload=None,
            selection_metrics={"selection_reason": "test"},
        )
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            save_calibration_artifact_v5(
                handle.name,
                model,
                gain_rates,
                strategy,
                0.8,
                {"evidence_adapter": "canonical"},
                {"gate": {"pass": True}, "undefined_metric": float("nan")},
            )
            loaded = load_calibration_artifact_v5(handle.name)
            self.assertEqual(loaded.k_strategy.fixed_k, 2)
            self.assertEqual(loaded.binding_threshold, 0.8)

            handle.seek(0)
            payload = json.load(handle)
            self.assertEqual(payload["artifact_version"], ARTIFACT_VERSION)
            self.assertIsNone(payload["metrics"]["undefined_metric"])
            handle.seek(0)
            self.assertNotIn("NaN", handle.read())
            self.assertNotIn("label_schema", payload)
            payload["supervision_standard"]["target"] = "title hit"
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(ValueError):
                load_calibration_artifact_v5(handle.name)

    def test_monotonic_calibrator_reaches_constrained_optimum(self):
        examples = []
        for index in range(80):
            support = 0.05 + 0.90 * ((index * 17) % 79) / 78
            novelty = 0.10 + 0.85 * ((index * 29) % 77) / 76
            is_gain = support * novelty > 0.30
            examples.append(
                MarginalUtilityExample(
                    "DECOMPOSE", "tgt", False, support, novelty, is_gain, float(is_gain)
                )
            )
        calibrator = MonotonicMarginalCalibrator.fit(examples, max_iterations=200)
        self.assertGreaterEqual(calibrator.support_slope, 0.0)
        self.assertGreaterEqual(calibrator.novelty_slope, 0.0)

        y = np.asarray([float(example.is_gain) for example in examples])
        support = np.asarray(
            [np.clip(np.log(example.support_score / (1.0 - example.support_score)), -12, 12)
             for example in examples]
        )
        novelty = np.asarray(
            [np.clip(np.log(example.novelty_score / (1.0 - example.novelty_score)), -12, 12)
             for example in examples]
        )
        design = np.column_stack((np.ones_like(support), support, novelty))
        beta = np.asarray(
            [calibrator.intercept, calibrator.support_slope, calibrator.novelty_slope]
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        gradient = design.T @ (probabilities - y)
        gradient[1:] += 1e-2 * beta[1:]
        self.assertLess(abs(float(gradient[0])), 1e-4)
        for coefficient, component in zip(beta[1:], gradient[1:]):
            if coefficient > 1e-8:
                self.assertLess(abs(float(component)), 1e-4)
            else:
                self.assertGreaterEqual(float(component), -1e-4)

    def test_quality_metrics_expose_novelty_shortcut_and_nonrepeat_subset(self):
        examples = [
            MarginalUtilityExample("EXPAND", "tgt", False, 0.9, 1.0, True, 0.5),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.2, 1.0, False, 0.0),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.9, 0.0, False, 0.0),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.1, 0.0, False, 0.0),
        ] * 8
        model, gain_rates, _ = fit_observation_model_v5(examples)
        metrics = posterior_quality_metrics_v5(examples, model, gain_rates)
        self.assertIn("novelty_only_auc", metrics)
        self.assertIn("posterior_auc_gain_over_novelty", metrics)
        self.assertEqual(metrics["nonrepeat"]["n"], 16)
        self.assertGreater(metrics["nonrepeat"]["raw_support_auc"], 0.5)

    def test_builder_recovers_annotation_ids_and_groups_duplicate_questions(self):
        builder = self._builder_module()
        question = "Where was Alice born?"
        annotation = {
            "_id": "hotpot-q1",
            "question": question,
            "context": {
                "title": ["Alice"],
                "sentences": [["Alice was born in Paris."]],
            },
        }
        records = builder._join_annotations(
            [{"problem": question}, {"problem": question}], [annotation]
        )
        self.assertTrue(all(builder._record_id(record) == "hotpot-q1" for record in records))
        self.assertTrue(
            all(record["_question_id_source"] == "annotation_question_join" for record in records)
        )
        entries = builder._annotation_context_entries(records[0])
        enriched = builder._enrich_document(
            "Alice: Alice was born in Paris.", entries
        )
        self.assertEqual(enriched["document_id"], "hotpot-q1:context:0")
        self.assertEqual(enriched["document_id_source"], "annotation_context")

        hash_joined = builder._join_annotations(
            [{"problem": "Who is Bob?"}],
            [{"question": "Who is Bob?", "context": {"title": [], "sentences": []}}],
        )[0]
        self.assertTrue(
            builder._record_id(hash_joined).startswith("annotation_question_sha256:")
        )
        self.assertEqual(
            hash_joined["_question_id_source"],
            "annotation_question_sha256_join",
        )

        extra = builder._join_annotations(
            [{"id": f"q{index}", "problem": f"question {index}"} for index in range(8)],
            [],
        )
        fit, validation, test = builder._split_by_question_id(
            [*records, *extra], validation_frac=0.2, test_frac=0.2, seed=0
        )
        memberships = [
            any(builder._record_id(record) == "hotpot-q1" for record in split)
            for split in (fit, validation, test)
        ]
        self.assertEqual(sum(memberships), 1)

    def test_validity_gate_rejects_shortcut_and_missing_action(self):
        builder = self._builder_module()
        examples = [SimpleNamespace() for _ in range(20)]
        rows = [SimpleNamespace() for _ in range(100)]
        metrics = {
            "posterior_auc": 0.95,
            "average_precision": 0.90,
            "posterior_auc_gain_over_novelty": 0.01,
            "nonrepeat": {"n": 20, "raw_support_auc": 0.80},
        }
        action_counts = {
            "EXPAND": {"target_examples": 20},
            "REWRITE": {"target_examples": 0},
            "DECOMPOSE": {"target_examples": 20},
        }
        provenance = {
            "dataset_question_id_fraction": 1.0,
            "corpus_document_id_fraction": 1.0,
        }
        gate = builder.evaluate_validity_gate(
            examples,
            rows,
            metrics,
            action_counts,
            provenance,
            0.8,
            ("EXPAND", "REWRITE", "DECOMPOSE"),
            min_validation_examples=20,
            min_validation_fit_eligible_fraction=0.15,
            min_validation_posterior_auc=0.75,
            min_validation_average_precision=0.30,
            min_validation_posterior_gain_over_novelty=0.05,
            min_validation_nonrepeat_examples=20,
            min_validation_nonrepeat_raw_support_auc=0.70,
            min_fit_target_examples_per_action=10,
            min_dataset_question_id_fraction=1.0,
            min_corpus_document_id_fraction=0.95,
        )
        self.assertFalse(gate["pass"])
        self.assertIn("posterior_auc_gain_over_novelty", gate["failed_checks"])
        self.assertIn("fit_target_examples_REWRITE", gate["failed_checks"])

    def test_auto_k_uses_fixed_for_constant_requirement_count(self):
        fit = [KExample(np.asarray([1.0, 0.0]), 2) for _ in range(120)]
        validation = [KExample(np.asarray([1.0, 0.0]), 2) for _ in range(20)]
        strategy = select_k_strategy_v5("auto", fit, validation, k_max=4)
        self.assertEqual(strategy.mode, "fixed")
        self.assertEqual(strategy.fixed_k, 2)
        self.assertIn("constant_k", strategy.selection_metrics["selection_reason"])

    def test_binding_threshold_is_selected_by_precision(self):
        examples = [
            MarginalUtilityExample("EXPAND", "tgt", False, 0.9, 1.0, True, 0.5),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.8, 1.0, True, 0.5),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.7, 1.0, False, 0.0),
            MarginalUtilityExample("EXPAND", "tgt", False, 0.1, 1.0, False, 0.0),
        ]
        threshold, metrics = choose_binding_threshold_v5(
            examples, [0.95, 0.90, 0.70, 0.10], min_precision=1.0, min_predicted_gain=2
        )
        self.assertEqual(threshold, 0.90)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
