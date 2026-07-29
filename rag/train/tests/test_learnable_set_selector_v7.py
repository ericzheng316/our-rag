import unittest
from dataclasses import replace

import numpy as np

from belief.acec.answer_value_v7 import (
    AnswerValueExampleV7,
    fit_answer_value_crossfit_v7,
    fit_answer_value_v7,
)
from belief.acec.belief_simulator_v7 import BeliefSimulatorV7
from belief.acec.belief_state import ACECBeliefState
from belief.acec.contracts_v7 import (
    CandidateV7,
    SelectorStateV7,
    assert_sf_label_free,
)
from belief.acec.runtime_v7 import BeliefSelectorRuntimeV7
from belief.acec.set_selector_v7 import (
    SELECTOR_FEATURE_NAMES_V7,
    SequentialSetSelectorV7,
    initialize_selector_artifact_v7,
)


def _state():
    return SelectorStateV7(
        question_id="q",
        turn_index=1,
        coverage=0.25,
        coverage_std=0.2,
        k_entropy=0.4,
        slot_probabilities=(0.2, 0.3),
        slot_weights=(0.5, 0.5),
        slot_bound=(False, False),
        target_slot=0,
        retrieval_budget_remaining=4,
    )


def _candidate(candidate_id, rank, score, entailment, hits, contradiction=()):
    return CandidateV7(
        candidate_id=candidate_id,
        contents=f"Contents for {candidate_id}",
        retrieval_rank=rank,
        retrieval_score=score,
        slot_entailment=tuple(entailment),
        slot_hit_probabilities=tuple(hits),
        slot_contradiction=tuple(contradiction),
    )


class DummyEmbedder:
    def encode(self, texts, **kwargs):
        return np.asarray(
            [[len(text), sum(ord(char) for char in text) % 17 + 1] for text in texts],
            dtype=float,
        )


class DummyNLI:
    def score(self, premise, hypothesis):
        del hypothesis
        return 0.9 if "useful" in premise else 0.1


class LearnableSetSelectorV7Test(unittest.TestCase):
    def test_selector_contract_rejects_supporting_fact_labels(self):
        with self.assertRaisesRegex(ValueError, "forbidden SF label"):
            assert_sf_label_free({"supporting_facts": {"title": ["leak"]}})

    def test_simulator_uses_conditional_set_marginal_without_mutation(self):
        state = _state()
        candidates = (
            _candidate("a", 0, 2.0, (0.9, 0.1), (0.8, 0.1)),
            _candidate("b", 1, 1.0, (0.8, 0.9), (0.7, 0.9)),
        )
        simulator = BeliefSimulatorV7(state, candidates)
        base = simulator.base
        after_a = simulator.simulate(("a",))
        gain_b_after_a = simulator.conditional_metrics("b", ("a",))
        self.assertEqual(simulator.base, base)
        self.assertGreater(after_a.coverage, base.coverage)
        self.assertGreater(gain_b_after_a["conditional_coverage_gain"], 0.0)
        self.assertAlmostEqual(
            simulator.simulate(("a", "b")).coverage,
            gain_b_after_a["coverage_after"],
        )

    def test_semantic_contradiction_is_not_hard_filtered(self):
        state = _state()
        candidates = (
            _candidate("high_rel", 0, 4.0, (0.2, 0.2), (0.2, 0.2)),
            _candidate(
                "contradictory",
                1,
                1.0,
                (0.8, 0.8),
                (0.8, 0.8),
                contradiction=(0.99, 0.99),
            ),
        )
        selector = SequentialSetSelectorV7(
            artifact=None, strategy="fixed_rel_plus_coverage"
        )
        selected, trace = selector.select(state, candidates, k=2)
        self.assertEqual({candidate.candidate_id for candidate in selected}, {"high_rel", "contradictory"})
        self.assertEqual(set(trace.pool_ids), {"high_rel", "contradictory"})
        for step in trace.steps:
            self.assertAlmostEqual(sum(step.candidate_probabilities.values()), 1.0)
            self.assertTrue(
                all(
                    probability > 0.0
                    for probability in step.candidate_probabilities.values()
                )
            )

    def test_feature_ablation_is_frozen_into_runtime_artifact(self):
        artifact = initialize_selector_artifact_v7(
            fit_question_id_sha256="fit",
            validation_question_id_sha256="validation",
            seed=3,
        )
        ablated = replace(
            artifact,
            metadata={"disabled_features": ["conditional_coverage_gain"]},
        )
        baseline = {name: 0.0 for name in SELECTOR_FEATURE_NAMES_V7}
        changed = dict(baseline)
        changed["conditional_coverage_gain"] = 100.0
        self.assertEqual(ablated.score(baseline), ablated.score(changed))

    def test_answer_value_fit_rejects_question_leakage(self):
        fit = [AnswerValueExampleV7("same", {"x": 1.0}, 0.2)]
        validation = [AnswerValueExampleV7("same", {"x": 2.0}, -0.1)]
        with self.assertRaisesRegex(ValueError, "leaks"):
            fit_answer_value_v7(fit, validation)

    def test_answer_value_pairwise_metric_is_within_state(self):
        fit = [
            AnswerValueExampleV7(
                "fit_q", {"x": 0.0}, 0.0, state_id="fit_state"
            ),
            AnswerValueExampleV7(
                "fit_q", {"x": 1.0}, 1.0, state_id="fit_state"
            ),
        ]
        validation = [
            AnswerValueExampleV7(
                "validation_q",
                {"x": 0.0},
                0.0,
                state_id="validation_state",
            ),
            AnswerValueExampleV7(
                "validation_q",
                {"x": 1.0},
                1.0,
                state_id="validation_state",
            ),
        ]
        artifact = fit_answer_value_v7(fit, validation, l2=0.1)
        self.assertEqual(
            artifact.metrics["validation_pairwise_pairs"], 1.0
        )
        self.assertEqual(
            artifact.metrics["validation_pairwise_accuracy"], 1.0
        )

    def test_answer_value_crossfit_uses_out_of_fold_artifact(self):
        fit = [
            AnswerValueExampleV7(
                f"q{index}", {"x": float(index)}, float(index) / 10.0
            )
            for index in range(40)
        ]
        validation = [
            AnswerValueExampleV7(
                f"v{index}", {"x": float(index)}, float(index) / 10.0
            )
            for index in range(10)
        ]
        bundle = fit_answer_value_crossfit_v7(
            fit, validation, folds=5, l2=1.0
        )
        self.assertEqual(len(bundle.fold_artifacts), 5)
        self.assertIsNot(
            bundle.artifact_for_question("q0"), bundle.full_artifact
        )
        self.assertIs(
            bundle.artifact_for_question("unseen"), bundle.full_artifact
        )

    def test_runtime_selects_before_single_live_update_and_matches_simulator(self):
        belief = ACECBeliefState(DummyEmbedder(), DummyNLI())
        belief.reset("Which evidence is useful?")
        selector = SequentialSetSelectorV7(artifact=None, strategy="relevance")
        runtime = BeliefSelectorRuntimeV7(selector, selected_k=1)
        documents = [
            {"id": "useful", "contents": "useful evidence", "score": 2.0},
            {"id": "other", "contents": "other evidence", "score": 1.0},
        ]
        result = runtime.turn(
            question_id="q",
            belief=belief,
            query="find useful evidence",
            candidate_documents=documents,
            retrieval_budget_remaining=4,
        )
        self.assertEqual(
            result.selection_trace.selected_ids,
            ("useful",),
        )
        self.assertEqual(belief.coverage_belief.turn, 1)
        self.assertLessEqual(result.simulator_live_coverage_error, 1e-6)
        self.assertGreater(result.belief_result.delta_coverage, 0.0)

        second_belief = ACECBeliefState(DummyEmbedder(), DummyNLI())
        second_belief.reset("Which evidence is useful?")
        expanded = runtime.turn(
            question_id="q2",
            belief=second_belief,
            query="find useful evidence",
            candidate_documents=documents,
            retrieval_budget_remaining=4,
            selected_k=2,
        )
        self.assertEqual(len(expanded.selection_trace.selected_ids), 2)
        self.assertEqual(expanded.selection_trace.selected_k, 2)
        self.assertEqual(runtime.selected_k, 1)


if __name__ == "__main__":
    unittest.main()
