import unittest

from belief.acec.evidence_contract_v63 import (
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    EvidenceSelectionV63,
    EvidenceTraceV63,
    EvidenceTurnV63,
    EvidenceUnitV63,
    assert_runtime_payload_has_no_gold,
)
from short_answer_contract_v63 import (
    ANSWER_CONTRACT_VERSION,
    parse_tagged_short_answer,
)


def _native_unit(evidence_id="q1:d0:s0"):
    return EvidenceUnitV63(
        evidence_id=evidence_id,
        source_id="Title",
        unit_index=0,
        text="A native sentence.",
        document_id="q1:d0",
        retrieval_turn=0,
        retrieval_rank=0,
        scores={"raw_answer_nli": 0.8},
    )


class EvidenceContractV63Test(unittest.TestCase):
    def test_trace_round_trip_preserves_order(self):
        unit = _native_unit()
        trace = EvidenceTraceV63(
            question_id="q1",
            question="What is the answer?",
            draft_answer="answer",
            candidates=(unit,),
            turns=(EvidenceTurnV63(0, "answer evidence", (unit.evidence_id,)),),
            answer_contract_version=ANSWER_CONTRACT_VERSION,
        )
        restored = EvidenceTraceV63.from_dict(trace.to_dict())
        self.assertEqual(restored, trace)
        self.assertEqual(restored.ordered_candidates(), (unit,))
        self.assertEqual(restored.schema, EVIDENCE_SCHEMA)
        self.assertEqual(restored.schema_version, EVIDENCE_SCHEMA_VERSION)

    def test_repeated_retrieval_reference_is_deduplicated_in_candidate_order(self):
        unit = _native_unit()
        trace = EvidenceTraceV63(
            question_id="q1",
            question="Question?",
            draft_answer="answer",
            candidates=(unit,),
            turns=(
                EvidenceTurnV63(0, "first query", (unit.evidence_id,)),
                EvidenceTurnV63(1, "repeat query", (unit.evidence_id,)),
            ),
            answer_contract_version=ANSWER_CONTRACT_VERSION,
        )
        self.assertEqual(trace.ordered_candidates(), (unit,))

    def test_runtime_payload_fails_closed_on_gold_fields(self):
        for payload in (
            {"supporting_facts": [["Title", 0]]},
            {"metadata": {"gold_answer": "answer"}},
            {"nested": [{"sf_label": 1}]},
            {"answer": "labeled answer"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "leaks evaluator field"):
                    assert_runtime_payload_has_no_gold(payload)

    def test_unmappable_evidence_is_not_exported_as_official_sp(self):
        native = _native_unit()
        unmappable = EvidenceUnitV63(
            evidence_id="q1:web",
            source_id=None,
            unit_index=None,
            text="An open-web sentence.",
            document_id="web-doc",
            retrieval_turn=0,
            retrieval_rank=1,
            provenance_status="unmappable",
        )
        trace = EvidenceTraceV63(
            question_id="q1",
            question="Question?",
            draft_answer="answer",
            candidates=(native, unmappable),
            turns=(
                EvidenceTurnV63(
                    0, "query", (native.evidence_id, unmappable.evidence_id)
                ),
            ),
            answer_contract_version=ANSWER_CONTRACT_VERSION,
        )
        selection = EvidenceSelectionV63(
            selector_name="test",
            selector_version="test",
            selected_evidence_ids=(native.evidence_id, unmappable.evidence_id),
            score_key="raw_answer_nli",
            stop_mode="test",
            stop_value=2,
        )
        self.assertEqual(selection.official_hotpot_sp(trace), (("Title", 0),))


class ShortAnswerContractV63Test(unittest.TestCase):
    def test_parser_extracts_only_one_tagged_short_span(self):
        parsed = parse_tagged_short_answer(
            "Reasoning can precede it.\n<answer>  Arthur's Magazine  </answer>"
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.answer, "Arthur's Magazine")
        self.assertEqual(parsed.contract_version, ANSWER_CONTRACT_VERSION)

    def test_parser_rejects_ambiguous_or_missing_answers(self):
        cases = [
            "The answer is swimming.",
            "<answer></answer>",
            "<answer>a</answer><answer>b</answer>",
            "<answer><answer>a</answer></answer>",
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(parse_tagged_short_answer(value).valid)


if __name__ == "__main__":
    unittest.main()
