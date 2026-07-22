import json
from pathlib import Path
import tempfile
import unittest

from eval_r3_processed_answers_api import (
    compute_r3_em_f1,
    extract_candidate_answers_strict,
    load_inputs,
    paired_comparison,
    request_candidate_answers,
    score_prediction,
    summarize_variant,
)


class R3ProcessedAnswersApiTest(unittest.TestCase):
    def test_candidate_parser_matches_official_contract(self):
        self.assertEqual(
            extract_candidate_answers_strict('Candidates: ["George Orwell", "Eric Blair"]'),
            ["George Orwell", "Eric Blair"],
        )
        self.assertEqual(
            extract_candidate_answers_strict("answer=[1949, '1949-10-01']"),
            ["1949", "1949-10-01"],
        )
        self.assertEqual(extract_candidate_answers_strict("[]"), [])
        self.assertIsNone(extract_candidate_answers_strict("not a list"))
        self.assertIsNone(extract_candidate_answers_strict("[['nested']]"))

    def test_processed_metric_uses_any_candidate_and_max_f1(self):
        em, f1 = compute_r3_em_f1(
            ["George Orwell"],
            ["Eric Arthur Blair", "George Orwell"],
        )
        self.assertEqual(em, 1.0)
        self.assertEqual(f1, 1.0)
        em, f1 = compute_r3_em_f1(["George Orwell"], ["George Orwell writer"])
        self.assertEqual(em, 0.0)
        self.assertAlmostEqual(f1, 0.8)

    def test_score_short_circuit_and_extracted_answer(self):
        manifest = {
            "question": "Who wrote 1984?",
            "golden_answers": ["George Orwell"],
        }
        exact = score_prediction(
            variant="base",
            qid="q1",
            manifest_row=manifest,
            prediction_row={
                "prediction": "The George Orwell.",
                "golden_answers": ["George Orwell"],
                "em": 1.0,
                "f1": 1.0,
                "answered": True,
            },
            extraction=None,
        )
        self.assertEqual(exact["status"], "raw_exact_short_circuit")
        self.assertEqual(exact["r3_processed_em"], 1.0)

        verbose = score_prediction(
            variant="base",
            qid="q1",
            manifest_row=manifest,
            prediction_row={
                "prediction": "The novel was written by George Orwell.",
                "golden_answers": ["George Orwell"],
                "em": 0.0,
                "f1": 0.5,
                "answered": True,
            },
            extraction={
                "ok": True,
                "candidates": ["George Orwell"],
                "raw_output": '["George Orwell"]',
                "attempts": 1,
                "error_type": "",
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )
        self.assertEqual(verbose["status"], "extracted")
        self.assertEqual(verbose["r3_direct_em"], 0.0)
        self.assertEqual(verbose["r3_processed_em"], 1.0)

    def test_openai_compatible_request_contract(self):
        class Usage:
            prompt_tokens = 10
            completion_tokens = 3
            total_tokens = 13

        class Message:
            content = '["George Orwell"]'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            usage = Usage()

        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return Response()

        class Chat:
            def __init__(self):
                self.completions = Completions()

        class Client:
            def __init__(self):
                self.chat = Chat()

        client = Client()
        result = request_candidate_answers(
            client,
            model="Qwen2.5-72B-Instruct",
            question="Who wrote 1984?",
            answer="It was George Orwell.",
            max_tokens=512,
            retries=0,
            provider_routing=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"], ["George Orwell"])
        self.assertEqual(result["usage"]["total_tokens"], 13)
        self.assertEqual(client.chat.completions.kwargs["temperature"], 0.0)
        self.assertFalse(client.chat.completions.kwargs["stream"])
        self.assertIn("provider", client.chat.completions.kwargs["extra_body"])

    def test_manifest_prediction_integrity_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            manifest.write_text(
                json.dumps(
                    {"id": "q1", "question": "Q?", "golden_answers": ["A"]}
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "prediction": "A",
                        "golden_answers": ["A"],
                        "em": 1.0,
                        "f1": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            order, loaded_manifest, loaded_predictions = load_inputs(
                manifest, [("base", predictions)]
            )
            self.assertEqual(order, ["q1"])
            self.assertEqual(loaded_manifest["q1"]["question"], "Q?")
            self.assertEqual(loaded_predictions["base"]["q1"]["prediction"], "A")

    def test_summary_and_paired_bootstrap(self):
        def record(qid, hotpot_em, processed_em):
            return {
                "id": qid,
                "answered": True,
                "input_hotpot_em": hotpot_em,
                "input_hotpot_f1": hotpot_em,
                "r3_direct_em": hotpot_em,
                "r3_direct_f1": hotpot_em,
                "r3_processed_em": processed_em,
                "r3_processed_f1": processed_em,
                "status": "raw_exact_short_circuit" if hotpot_em else "extracted",
                "candidate_answers": ["x"] if not hotpot_em else [],
                "extractor_attempts": 0 if hotpot_em else 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        baseline = [record("q1", 0.0, 0.0), record("q2", 1.0, 1.0)]
        candidate = [record("q1", 1.0, 1.0), record("q2", 1.0, 1.0)]
        summary = summarize_variant(candidate)
        self.assertEqual(summary["r3_processed_em"], 1.0)
        comparison = paired_comparison(
            baseline,
            candidate,
            bootstrap_samples=100,
            seed=7,
        )
        self.assertEqual(comparison["r3_processed_em"]["delta"], 0.5)
        self.assertEqual(comparison["r3_processed_em"]["wins"], 1)
        self.assertEqual(comparison["r3_processed_em"]["losses"], 0)


if __name__ == "__main__":
    unittest.main()
