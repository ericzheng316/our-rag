import unittest

from belief.acec.evidence_contract_v63 import (
    EvidenceTraceV63,
    EvidenceTurnV63,
    EvidenceUnitV63,
)
from belief.acec.evidence_selector_minimal_v63 import (
    SharedStoppingRuleV63,
    answer_hypothesis,
    select_acec_existing_v63,
    select_generic_nli_v63,
)
from short_answer_contract_v63 import ANSWER_CONTRACT_VERSION


def _unit(evidence_id, title, score, accepted, rank):
    return EvidenceUnitV63(
        evidence_id=evidence_id,
        source_id=title,
        unit_index=0,
        text=f"Evidence from {title}.",
        document_id=f"doc:{title}",
        retrieval_turn=0,
        retrieval_rank=rank,
        scores={
            "raw_answer_nli": score,
            "acec_document_accepted": float(accepted),
        },
    )


def _trace():
    distractor = _unit("wrong", "Wrong", 0.99, False, 0)
    support = _unit("right", "Right", 0.80, True, 1)
    return EvidenceTraceV63(
        question_id="q",
        question="Who is right?",
        draft_answer="Right",
        candidates=(distractor, support),
        turns=(
            EvidenceTurnV63(0, "right evidence", (distractor.evidence_id, support.evidence_id)),
        ),
        answer_contract_version=ANSWER_CONTRACT_VERSION,
    )


class MinimalSelectorV63Test(unittest.TestCase):
    def test_same_gold_k_rule_is_used_but_acec_filters_documents(self):
        trace = _trace()
        rule = SharedStoppingRuleV63(mode="gold_cardinality_oracle")
        generic = select_generic_nli_v63(trace, rule, gold_cardinality=1)
        acec = select_acec_existing_v63(trace, rule, gold_cardinality=1)
        self.assertEqual(generic.stop_mode, acec.stop_mode)
        self.assertEqual(generic.selected_evidence_ids, ("wrong",))
        self.assertEqual(acec.selected_evidence_ids, ("right",))

    def test_same_threshold_is_shared_without_posterior_rescoring(self):
        trace = _trace()
        rule = SharedStoppingRuleV63(mode="shared_threshold", threshold=0.75)
        generic = select_generic_nli_v63(trace, rule)
        acec = select_acec_existing_v63(trace, rule)
        self.assertEqual(generic.stop_value, acec.stop_value)
        self.assertEqual(generic.selected_evidence_ids, ("wrong", "right"))
        self.assertEqual(acec.selected_evidence_ids, ("right",))
        self.assertEqual(acec.score_key, "raw_answer_nli")

    def test_missing_acec_replay_fails_closed(self):
        trace = _trace()
        broken_unit = trace.candidates[0].with_scores({})
        # with_scores cannot remove a key, so construct the malformed candidate.
        broken_unit = EvidenceUnitV63(
            evidence_id=broken_unit.evidence_id,
            source_id=broken_unit.source_id,
            unit_index=broken_unit.unit_index,
            text=broken_unit.text,
            document_id=broken_unit.document_id,
            retrieval_turn=broken_unit.retrieval_turn,
            retrieval_rank=broken_unit.retrieval_rank,
            scores={"raw_answer_nli": 0.99},
        )
        broken = trace.with_candidates((broken_unit, trace.candidates[1]))
        with self.assertRaisesRegex(ValueError, "lacks ACEC document acceptance"):
            select_acec_existing_v63(
                broken, SharedStoppingRuleV63("shared_threshold", threshold=0.5)
            )

    def test_answer_hypothesis_is_deterministic(self):
        self.assertEqual(
            answer_hypothesis("Who is right?", "Right"),
            'The answer to "Who is right?" is "Right".',
        )


if __name__ == "__main__":
    unittest.main()
