from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from belief.acec.answer_verifier_v64 import (
    TrajectoryCandidateV64,
    VerifierExampleV64,
    evaluate_best_of_n_v64,
    evaluate_policy_controller_factorial_v64,
    fit_answer_verifier_v64,
    load_answer_verifier_artifact_v64,
    paired_bootstrap_delta_v64,
    save_answer_verifier_artifact_v64,
    select_candidate_v64,
)


def _candidate(
    qid: str,
    index: int,
    answer: str,
    *,
    grounding: float,
    coverage: float,
    std: float = 0.1,
    retrievals: int = 2,
) -> TrajectoryCandidateV64:
    return TrajectoryCandidateV64(
        question_id=qid,
        sample_index=index,
        answer=answer,
        grounding_score=grounding,
        coverage=coverage,
        coverage_std=std,
        retrieval_calls=retrievals,
    )


class AnswerVerifierV64Test(unittest.TestCase):
    def test_oracle_at_n_separates_sampling_headroom_from_selection(self):
        candidates = [
            _candidate("q1", 0, "wrong", grounding=0.9, coverage=0.9),
            _candidate("q1", 1, "right", grounding=0.2, coverage=0.2),
            _candidate("q2", 0, "also wrong", grounding=0.8, coverage=0.8),
            _candidate("q2", 1, "correct", grounding=0.7, coverage=0.7),
        ]
        result = evaluate_best_of_n_v64(
            candidates,
            {"q1": ["right"], "q2": ["correct"]},
            k_values=[1, 2],
            modes=["first"],
        )
        self.assertEqual(result["summaries"]["first@1"]["em"], 0.0)
        self.assertEqual(result["summaries"]["first@2"]["oracle_em"], 1.0)
        self.assertEqual(
            result["summaries"]["first@2"]["selection_regret_em"], 1.0
        )

    def test_majority_nli_and_acec_are_distinct_selectors(self):
        candidates = [
            _candidate("q", 0, "majority", grounding=0.4, coverage=0.9),
            _candidate("q", 1, "majority", grounding=0.3, coverage=0.9),
            _candidate("q", 2, "nli", grounding=0.95, coverage=0.1),
            _candidate("q", 3, "acec", grounding=0.8, coverage=0.95),
        ]
        self.assertEqual(
            select_candidate_v64(candidates, mode="majority").answer, "majority"
        )
        self.assertEqual(
            select_candidate_v64(candidates, mode="nli").answer, "nli"
        )
        self.assertEqual(
            select_candidate_v64(candidates, mode="acec_zero_shot").answer,
            "acec",
        )

    def test_fit_is_question_disjoint_and_runtime_gated(self):
        fit_candidates = [
            _candidate("fit1", 0, "right", grounding=0.95, coverage=0.95),
            _candidate("fit1", 1, "wrong", grounding=0.05, coverage=0.05),
            _candidate("fit2", 0, "right", grounding=0.90, coverage=0.90),
            _candidate("fit2", 1, "wrong", grounding=0.10, coverage=0.10),
        ]
        validation_candidates = [
            _candidate("val1", 0, "right", grounding=0.95, coverage=0.90),
            _candidate("val1", 1, "wrong", grounding=0.10, coverage=0.20),
            _candidate("val2", 0, "wrong", grounding=0.15, coverage=0.15),
            _candidate("val2", 1, "right", grounding=0.90, coverage=0.95),
        ]
        fit = [
            VerifierExampleV64.from_candidate(
                candidate, correct=index % 2 == 0
            )
            for index, candidate in enumerate(fit_candidates)
        ]
        validation = [
            VerifierExampleV64.from_candidate(
                candidate, correct=candidate.answer == "right"
            )
            for candidate in validation_candidates
        ]
        artifact = fit_answer_verifier_v64(
            fit, validation, steps=500, learning_rate=0.2
        )
        self.assertEqual(artifact.metrics["validation"]["auc"], 1.0)
        self.assertEqual(
            artifact.metrics["validation"]["top1_oracle_recovery"], 1.0
        )
        self.assertTrue(artifact.metrics["gate"]["pass"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verifier.json"
            save_answer_verifier_artifact_v64(path, artifact)
            loaded = load_answer_verifier_artifact_v64(path)
            self.assertEqual(loaded.to_dict(), artifact.to_dict())
            self.assertEqual(
                select_candidate_v64(
                    validation_candidates[:2],
                    mode="acec_calibrated",
                    artifact=loaded,
                ).answer,
                "right",
            )

        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_answer_verifier_v64(fit, fit)

    def test_factorial_keeps_policy_controller_arms_separate(self):
        arms = {
            "frozen_r3__none": [
                _candidate("q", 0, "wrong", grounding=0.5, coverage=0.5)
            ],
            "acec_ep50__belief_gap": [
                _candidate("q", 0, "right", grounding=0.8, coverage=0.8)
            ],
        }
        result = evaluate_policy_controller_factorial_v64(
            arms,
            {"q": ["right"]},
            k_values=[1],
            modes=["first"],
        )
        self.assertEqual(
            result["frozen_r3__none"]["summaries"]["first@1"]["em"], 0.0
        )
        self.assertEqual(
            result["acec_ep50__belief_gap"]["summaries"]["first@1"]["em"], 1.0
        )

    def test_paired_bootstrap_validates_iterations(self):
        result = paired_bootstrap_delta_v64(
            {"q1": 1.0, "q2": 0.0},
            {"q1": 0.0, "q2": 0.0},
            iterations=100,
            seed=3,
        )
        self.assertEqual(result["delta"], 0.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            paired_bootstrap_delta_v64(
                {"q": 1.0},
                {"q": 0.0},
                iterations=0,
            )


if __name__ == "__main__":
    unittest.main()
