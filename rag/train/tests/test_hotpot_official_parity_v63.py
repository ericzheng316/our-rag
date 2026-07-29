import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from belief.acec.datasets.hotpotqa_v63 import HotpotQADistractorAdapterV63
from belief.acec.evidence_contract_v63 import assert_runtime_payload_has_no_gold
from eval_hotpot_joint_v63 import (
    aggregate_reference_metrics,
    export_official_submission,
    reference_question_metrics,
    run_official_evaluator,
)
from belief.acec.evidence_contract_v63 import EvidenceTraceV63, EvidenceTurnV63
from short_answer_contract_v63 import ANSWER_CONTRACT_VERSION


def _record():
    return {
        "_id": "fixture-1",
        "question": "Who wrote the work?",
        "answer": "Ada Lovelace",
        "type": "bridge",
        "level": "easy",
        "context": {
            "title": ["Work", "Ada Lovelace"],
            "sentences": [
                ["The work was published in 1843.", "It was written by Ada Lovelace."],
                ["Ada Lovelace was an English mathematician."],
            ],
        },
        "supporting_facts": {
            "title": ["Work", "Ada Lovelace"],
            "sent_id": [1, 0],
        },
    }


class HotpotNativeProvenanceV63Test(unittest.TestCase):
    def test_all_sentences_round_trip_to_native_title_id_text(self):
        adapter = HotpotQADistractorAdapterV63()
        adapter.validate_round_trip(_record())
        units = adapter.evidence_units(_record())
        self.assertEqual(len(units), 3)
        self.assertEqual(units[1].official_hotpot_pair, ("Work", 1))
        self.assertEqual(units[1].text, "It was written by Ada Lovelace.")
        self.assertEqual(units[2].official_hotpot_pair, ("Ada Lovelace", 0))

    def test_runtime_and_gold_manifests_are_separated(self):
        adapter = HotpotQADistractorAdapterV63()
        runtime = adapter.runtime_record(_record())
        evaluator = adapter.evaluator_record(_record())
        assert_runtime_payload_has_no_gold(runtime)
        self.assertNotIn("answer", runtime)
        self.assertNotIn("supporting_facts", runtime)
        self.assertEqual(evaluator["answer"], "Ada Lovelace")
        self.assertEqual(
            {tuple(value) for value in evaluator["supporting_facts"]},
            {("Work", 1), ("Ada Lovelace", 0)},
        )

    def test_preprocessor_writes_separate_immutable_artifacts(self):
        repo_root = Path(__file__).resolve().parents[3]
        program = repo_root / "run_scripts" / "prep_hotpot_distractor_provenance_v63.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dev.jsonl"
            source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            runtime = root / "runtime.jsonl"
            evaluator = root / "gold.json"
            manifest = root / "manifest.json"
            command = [
                sys.executable,
                str(program),
                "--input",
                str(source),
                "--runtime_output",
                str(runtime),
                "--evaluator_output",
                str(evaluator),
                "--manifest_output",
                str(manifest),
                "--split_name",
                "dev_fixture",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            runtime_row = json.loads(runtime.read_text())
            assert_runtime_payload_has_no_gold(runtime_row)
            metadata = json.loads(manifest.read_text())
            self.assertEqual(metadata["native_provenance_coverage"], 1.0)
            self.assertEqual(metadata["native_sentence_count"], 3)
            repeated = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("refusing to overwrite", repeated.stderr)


class HotpotMetricParityV63Test(unittest.TestCase):
    def test_joint_is_per_question_intersection_not_product_of_means(self):
        first = reference_question_metrics(
            "Ada Lovelace",
            {("Wrong", 0)},
            "Ada Lovelace",
            {("Work", 1)},
        )
        second = reference_question_metrics(
            "wrong answer",
            {("Work", 1)},
            "Ada Lovelace",
            {("Work", 1)},
        )
        aggregate = aggregate_reference_metrics([first, second])
        self.assertEqual(aggregate["em"], 0.5)
        self.assertEqual(aggregate["sp_em"], 0.5)
        self.assertEqual(aggregate["joint_em"], 0.0)
        self.assertNotEqual(
            aggregate["joint_em"], aggregate["em"] * aggregate["sp_em"]
        )

    def test_exporter_deduplicates_with_exact_set_semantics(self):
        submission = export_official_submission(
            {"q": "answer"},
            {"q": [("B", 1), ("A", 0), ("B", 1)]},
        )
        self.assertEqual(submission["sp"]["q"], [["A", 0], ["B", 1]])

    @unittest.skipUnless(
        os.environ.get("HOTPOT_OFFICIAL_EVALUATOR"),
        "set HOTPOT_OFFICIAL_EVALUATOR to run unchanged-script parity",
    )
    def test_reference_matches_unchanged_official_evaluator(self):
        evaluator = Path(os.environ["HOTPOT_OFFICIAL_EVALUATOR"])
        source_record = _record()
        record = HotpotQADistractorAdapterV63().evaluator_record(source_record)
        prediction = export_official_submission(
            {record["_id"]: "Ada Lovelace"},
            {record["_id"]: [("Work", 1), ("Ada Lovelace", 0)]},
        )
        reference = reference_question_metrics(
            "Ada Lovelace",
            {("Work", 1), ("Ada Lovelace", 0)},
            "Ada Lovelace",
            {("Work", 1), ("Ada Lovelace", 0)},
        )
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "prediction.json"
            gold_path = Path(directory) / "gold.json"
            prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
            gold_path.write_text(json.dumps([record]), encoding="utf-8")
            official = run_official_evaluator(evaluator, prediction_path, gold_path)
        for key in set(reference) & set(official):
            self.assertAlmostEqual(reference[key], official[key])

    @unittest.skipUnless(
        os.environ.get("HOTPOT_OFFICIAL_EVALUATOR"),
        "set HOTPOT_OFFICIAL_EVALUATOR to run full evaluator integration",
    )
    def test_full_fixed_trace_evaluator_uses_official_script(self):
        adapter = HotpotQADistractorAdapterV63()
        source_record = _record()
        raw_units = adapter.evidence_units(source_record)
        score_by_pair = {
            ("Work", 0): 0.1,
            ("Work", 1): 0.9,
            ("Ada Lovelace", 0): 0.8,
        }
        units = tuple(
            unit.with_scores(
                {
                    "raw_answer_nli": score_by_pair[unit.official_hotpot_pair],
                    "acec_document_accepted": 1.0,
                }
            )
            for unit in raw_units
        )
        repo_root = Path(__file__).resolve().parents[3]
        evaluator_program = repo_root / "rag" / "train" / "eval_hotpot_joint_v63.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "acec.json"
            artifact.write_text('{"artifact_version": 5}\n', encoding="utf-8")
            import hashlib

            artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            trace = EvidenceTraceV63(
                question_id="fixture-1",
                question=source_record["question"],
                draft_answer="Ada Lovelace",
                candidates=units,
                turns=(
                    EvidenceTurnV63(
                        0, "Who wrote the work?", tuple(unit.evidence_id for unit in units)
                    ),
                ),
                answer_contract_version=ANSWER_CONTRACT_VERSION,
                metadata={
                    "acec_artifact_sha256": artifact_hash,
                    "acec_artifact_version": 5,
                    "acec_binding_threshold": 0.75,
                    "acec_k_mode": "fixed",
                    "acec_replay_version": "fixture",
                    "provenance_adapter": adapter.name,
                    "provenance_adapter_version": adapter.version,
                    "trace_generator": "fixture",
                    "raw_nli_model_version": "fixture-nli",
                    "raw_nli_entailment_index": 1,
                },
            )
            trace_path = root / "trace.jsonl"
            trace_path.write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
            gold_path = root / "gold.json"
            gold_path.write_text(
                json.dumps([adapter.evaluator_record(source_record)]), encoding="utf-8"
            )
            output_dir = root / "evaluation"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(evaluator_program),
                    "--trace",
                    str(trace_path),
                    "--gold",
                    str(gold_path),
                    "--official_evaluator",
                    os.environ["HOTPOT_OFFICIAL_EVALUATOR"],
                    "--acec_artifact",
                    str(artifact),
                    "--output_dir",
                    str(output_dir),
                    "--shared_threshold",
                    "0.5",
                    "--threshold_provenance",
                    "frozen_unsupervised",
                    "--nli_model_version",
                    "fixture-nli",
                    "--bootstrap_iterations",
                    "10",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = json.loads((output_dir / "results.json").read_text())
            self.assertEqual(
                results["official_metrics"]["acec_shared_threshold"]["joint_em"],
                1.0,
            )
            self.assertEqual(
                results["official_reference_parity"]["acec_shared_threshold"][
                    "max_abs_error"
                ],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
