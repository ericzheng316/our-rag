import json
from pathlib import Path
import sys
import tempfile
import unittest

import assign_question_splits_v7
import build_counterfactual_sets_v7
import compare_selector_delta_c_ablation_v7
import eval_candidate_headroom_v7
import eval_selected_calibration_v7
import infer_belief_selector_v7
import summarize_selector_eval_v7
import validate_smoke_v7

from belief.acec.contracts_v7 import CandidateV7, SelectorStateV7
from belief.acec.set_selector_v7 import SequentialSetSelectorV7


def _run_main(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *arguments]
        module.main()
    finally:
        sys.argv = previous


class V7OfflinePipelineTest(unittest.TestCase):
    def test_title_headroom_cannot_exceed_one_with_duplicate_chunks(self):
        candidates = [
            {"metadata": {"title": "Gold"}},
            {"metadata": {"title": "Gold"}},
        ]
        self.assertEqual(
            eval_candidate_headroom_v7._title_recall(candidates, {"gold"}),
            1.0,
        )

    def test_split_counterfactual_and_inference_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            split = root / "split.jsonl"
            slates = root / "slates.jsonl"
            inferred = root / "inferred.jsonl"
            learned = root / "learned.jsonl"
            oracle = root / "oracle.jsonl"
            calibration = root / "calibration.json"
            selector_summary = root / "selector_summary.json"
            no_delta_c = root / "no_delta_c.jsonl"
            ablation_summary = root / "ablation_summary.json"
            smoke_metrics = root / "smoke_metrics.jsonl"
            smoke_gate = root / "smoke_gate.json"
            with raw.open("w", encoding="utf-8") as handle:
                for index in range(100):
                    state = SelectorStateV7(
                        question_id=f"q{index}",
                        turn_index=1,
                        coverage=0.1,
                        coverage_std=0.2,
                        k_entropy=0.3,
                        slot_probabilities=(0.1, 0.1),
                        slot_weights=(0.5, 0.5),
                        slot_bound=(False, False),
                        target_slot=0,
                        retrieval_budget_remaining=4,
                        metadata={
                            "selection_capacity": 1 if index == 0 else 2
                        },
                    )
                    candidates = [
                        CandidateV7(
                            candidate_id=f"q{index}:d{rank}",
                            contents=f"document {rank}",
                            retrieval_rank=rank,
                            retrieval_score=float(2 - rank),
                            slot_entailment=(0.8 - rank * 0.2, 0.2 + rank * 0.5),
                            slot_hit_probabilities=(0.7 - rank * 0.2, 0.2 + rank * 0.6),
                        )
                        for rank in range(2)
                    ]
                    handle.write(
                        json.dumps(
                            {
                                "state_id": f"s{index}",
                                "question": f"Question {index}?",
                                "gold_answers": [f"answer {index}"],
                                "state": state.to_dict(),
                                "candidates": [
                                    candidate.to_dict() for candidate in candidates
                                ],
                            }
                        )
                        + "\n"
                    )

            _run_main(
                assign_question_splits_v7,
                ["--input", str(raw), "--output", str(split)],
            )
            _run_main(
                build_counterfactual_sets_v7,
                [
                    "--input",
                    str(split),
                    "--output",
                    str(slates),
                    "--candidate_pool_size",
                    "2",
                    "--selected_k",
                    "2",
                ],
            )
            _run_main(
                infer_belief_selector_v7,
                [
                    "--input",
                    str(split),
                    "--output",
                    str(inferred),
                    "--strategy",
                    "relevance",
                    "--selected_k",
                    "2",
                ],
            )
            split_rows = [
                json.loads(line)
                for line in split.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["split"] for row in split_rows},
                {"fit", "validation", "test"},
            )
            slate_rows = [
                json.loads(line)
                for line in slates.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(slate_rows), 100)
            self.assertTrue(
                all(len(row["slates"]) == 4 for row in slate_rows)
            )
            inference_rows = [
                json.loads(line)
                for line in inferred.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(inference_rows), 100)
            self.assertTrue(
                len(inference_rows[0]["selected_documents"]) == 1
                and all(
                    len(row["selected_documents"]) == 2
                    for row in inference_rows[1:]
                )
            )

            selector = SequentialSetSelectorV7(
                artifact=None, strategy="relevance"
            )
            with learned.open("w", encoding="utf-8") as learned_handle, (
                oracle.open("w", encoding="utf-8")
            ) as oracle_handle:
                for row in slate_rows:
                    for slate in row["slates"]:
                        slate["answer_utility"] = 0.0
                    state = SelectorStateV7.from_dict(row["state"])
                    candidates = tuple(
                        CandidateV7.from_dict(value)
                        for value in row["candidates"]
                    )
                    _, trace = selector.select(state, candidates, k=1)
                    row["slates"].append(
                        {
                            "slate_id": "learned_v7",
                            "selected_ids": list(trace.selected_ids),
                            "selection_trace": trace.to_dict(),
                            "answer_utility": 1.0,
                        }
                    )
                    learned_handle.write(json.dumps(row) + "\n")
                    oracle_handle.write(
                        json.dumps(
                            {
                                "state_id": row["state_id"],
                                "gold_candidate_ids": [
                                    candidates[0].candidate_id
                                ],
                            }
                        )
                        + "\n"
                    )
            _run_main(
                eval_selected_calibration_v7,
                [
                    "--selection",
                    str(learned),
                    "--oracle",
                    str(oracle),
                    "--output",
                    str(calibration),
                ],
            )
            calibration_result = json.loads(
                calibration.read_text(encoding="utf-8")
            )
            self.assertEqual(
                calibration_result["calibrated_quantity"],
                "max_action_conditioned_slot_hit_probability",
            )
            self.assertEqual(calibration_result["selected_rows"], 100)
            _run_main(
                summarize_selector_eval_v7,
                [
                    "--input",
                    str(learned),
                    "--output",
                    str(selector_summary),
                    "--bootstrap_iterations",
                    "100",
                ],
            )
            summary = json.loads(selector_summary.read_text(encoding="utf-8"))
            self.assertTrue(summary["go"])
            self.assertEqual(summary["win_rate"], 1.0)
            with no_delta_c.open("w", encoding="utf-8") as handle:
                for line in learned.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    for slate in row["slates"]:
                        if slate["slate_id"] == "learned_v7":
                            slate["answer_utility"] = 0.0
                            slate["selection_trace"][
                                "selector_artifact_sha256"
                            ] = "ablated"
                    handle.write(json.dumps(row) + "\n")
            _run_main(
                compare_selector_delta_c_ablation_v7,
                [
                    "--full",
                    str(learned),
                    "--no_delta_c",
                    str(no_delta_c),
                    "--output",
                    str(ablation_summary),
                    "--bootstrap_iterations",
                    "100",
                ],
            )
            ablation = json.loads(
                ablation_summary.read_text(encoding="utf-8")
            )
            self.assertTrue(ablation["go"])
            self.assertEqual(ablation["win_rate_gap"], 1.0)
            smoke_metrics.write_text(
                json.dumps(
                    {
                        "schema": "acec_online_metrics_v7",
                        "episode": 1,
                        "format_error_rate": 0.0,
                        "selector_overhead_fraction_p95": 0.2,
                        "selection_trace_rate": 1.0,
                        "selector_calls_per_rollout": 1.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _run_main(
                validate_smoke_v7,
                [
                    "--input",
                    str(smoke_metrics),
                    "--output",
                    str(smoke_gate),
                ],
            )
            self.assertTrue(
                json.loads(smoke_gate.read_text(encoding="utf-8"))["pass"]
            )


if __name__ == "__main__":
    unittest.main()
