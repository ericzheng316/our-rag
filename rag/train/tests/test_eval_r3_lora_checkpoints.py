import json
import tempfile
import unittest

from eval_r3_lora_checkpoints import (
    exact_match_score,
    f1_score,
    load_heldout_questions,
    normalize_answer,
    parse_adapter,
    substring_match_score,
)


class HeldoutEvaluationHelpersTest(unittest.TestCase):
    def test_metric_normalization_matches_training_exact_match(self):
        self.assertEqual(normalize_answer("The Arthur's Magazine."), "arthurs magazine")
        self.assertEqual(exact_match_score("The answer", ["answer"]), 1.0)
        self.assertEqual(exact_match_score("Yes, because it is true", ["yes"]), 0.0)
        self.assertEqual(substring_match_score("Yes, because it is true", ["yes"]), 1.0)

    def test_hotpot_style_f1(self):
        self.assertEqual(f1_score("yes because", ["yes"]), 0.0)
        self.assertAlmostEqual(f1_score("Arthur's Magazine", ["Arthur's Magazine"]), 1.0)
        self.assertAlmostEqual(f1_score("Arthur Magazine", ["Arthur's Magazine"]), 0.5)
        self.assertAlmostEqual(f1_score("George Orwell writer", ["George Orwell"]), 0.8)

    def test_dev_metadata_supporting_fact_titles_are_loaded(self):
        row = {
            "id": "dev_1",
            "question": "Question?",
            "golden_answers": ["answer"],
            "metadata": {"supporting_facts": {"title": ["A", "B"], "sent_id": [0, 1]}},
        }
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            loaded = load_heldout_questions(handle.name, max_questions=1, seed=7)
        self.assertEqual(loaded[0]["id"], "dev_1")
        self.assertEqual(loaded[0]["sf_titles"], ["A", "B"])
        self.assertEqual(loaded[0]["golden_answers"].question_id, "dev_1")

    def test_adapter_parser(self):
        self.assertEqual(parse_adapter("step50=/tmp/step_50"), ("step50", "/tmp/step_50"))


if __name__ == "__main__":
    unittest.main()
