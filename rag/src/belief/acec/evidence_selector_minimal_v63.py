"""Minimal fixed-trace selectors for the ACEC v6.3 Gate-3 comparison.

Both arms rank native sentences with the same raw answer-conditioned NLI
score.  The ACEC arm differs only by filtering to documents accepted by the
existing document-level belief replay.  No document posterior is relabeled as
a calibrated sentence probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .evidence_contract_v63 import (
    EvidenceSelectionV63,
    EvidenceTraceV63,
    EvidenceUnitV63,
)


SELECTOR_VERSION = "minimal_fixed_trace_v63.1"
ANSWER_HYPOTHESIS_VERSION = "answer_hypothesis_v63.1"
RAW_NLI_SCORE_KEY = "raw_answer_nli"
ACEC_ACCEPT_SCORE_KEY = "acec_document_accepted"


def answer_hypothesis(question: str, answer: str) -> str:
    """Versioned deterministic H(q,a) shared by both selector arms."""

    clean_question = " ".join(str(question).split()).rstrip("?")
    clean_answer = " ".join(str(answer).split())
    if not clean_question or not clean_answer:
        raise ValueError("answer-conditioned NLI requires question and answer")
    return f'The answer to "{clean_question}?" is "{clean_answer}".'


@dataclass(frozen=True)
class SharedStoppingRuleV63:
    """One stopping rule instance must be passed to both selector arms."""

    mode: str
    threshold: Optional[float] = None
    max_units: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in {"shared_threshold", "gold_cardinality_oracle"}:
            raise ValueError(f"unsupported selector stop mode {self.mode!r}")
        if self.mode == "shared_threshold":
            if self.threshold is None or not 0.0 <= self.threshold <= 1.0:
                raise ValueError("shared_threshold mode requires threshold in [0, 1]")
        elif self.threshold is not None:
            raise ValueError("gold-cardinality oracle cannot also use a threshold")
        if self.max_units is not None and self.max_units <= 0:
            raise ValueError("max_units must be positive when set")

    def select_count(
        self,
        ranked: Sequence[EvidenceUnitV63],
        *,
        score_key: str,
        gold_cardinality: Optional[int],
    ) -> int:
        if self.mode == "gold_cardinality_oracle":
            if gold_cardinality is None or gold_cardinality < 0:
                raise ValueError("oracle selection requires non-negative gold_cardinality")
            count = int(gold_cardinality)
        else:
            assert self.threshold is not None
            count = sum(
                float(unit.scores[score_key]) >= self.threshold for unit in ranked
            )
        if self.max_units is not None:
            count = min(count, self.max_units)
        return min(count, len(ranked))

    @property
    def value(self) -> float:
        if self.mode == "shared_threshold":
            assert self.threshold is not None
            return float(self.threshold)
        return -1.0


def _rank(
    candidates: Iterable[EvidenceUnitV63], score_key: str
) -> List[EvidenceUnitV63]:
    units = list(candidates)
    missing = [unit.evidence_id for unit in units if score_key not in unit.scores]
    if missing:
        raise ValueError(f"candidates lack {score_key!r}: {missing[:5]}")
    for unit in units:
        score = float(unit.scores[score_key])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid {score_key!r} for {unit.evidence_id}")
    # Trace order is the stable final tie-break, so selectors cannot gain from
    # title/id lexical order and fixed-trace causality remains auditable.
    return [
        unit
        for _, unit in sorted(
            enumerate(units),
            key=lambda item: (-float(item[1].scores[score_key]), item[0]),
        )
    ]


def _selection(
    trace: EvidenceTraceV63,
    candidates: Sequence[EvidenceUnitV63],
    *,
    selector_name: str,
    stop_rule: SharedStoppingRuleV63,
    score_key: str,
    gold_cardinality: Optional[int],
    rejected_reasons: Dict[str, str],
) -> EvidenceSelectionV63:
    ranked = _rank(candidates, score_key)
    count = stop_rule.select_count(
        ranked, score_key=score_key, gold_cardinality=gold_cardinality
    )
    selected_ids = tuple(unit.evidence_id for unit in ranked[:count])
    reasons = dict(rejected_reasons)
    selected_set = set(selected_ids)
    for unit in ranked:
        reasons[unit.evidence_id] = (
            "selected_by_shared_raw_nli_ranking"
            if unit.evidence_id in selected_set
            else "below_shared_stop"
        )
    result = EvidenceSelectionV63(
        selector_name=selector_name,
        selector_version=SELECTOR_VERSION,
        selected_evidence_ids=selected_ids,
        score_key=score_key,
        stop_mode=stop_rule.mode,
        stop_value=(
            float(gold_cardinality)
            if stop_rule.mode == "gold_cardinality_oracle"
            and gold_cardinality is not None
            else stop_rule.value
        ),
        reasons=reasons,
    )
    result.validate_against(trace)
    return result


def select_generic_nli_v63(
    trace: EvidenceTraceV63,
    stop_rule: SharedStoppingRuleV63,
    *,
    score_key: str = RAW_NLI_SCORE_KEY,
    gold_cardinality: Optional[int] = None,
) -> EvidenceSelectionV63:
    mappable = [
        unit for unit in trace.ordered_candidates() if unit.official_hotpot_pair is not None
    ]
    rejected = {
        unit.evidence_id: "excluded_unmappable_provenance"
        for unit in trace.ordered_candidates()
        if unit.official_hotpot_pair is None
    }
    return _selection(
        trace,
        mappable,
        selector_name="raw_answer_conditioned_nli",
        stop_rule=stop_rule,
        score_key=score_key,
        gold_cardinality=gold_cardinality,
        rejected_reasons=rejected,
    )


def select_acec_existing_v63(
    trace: EvidenceTraceV63,
    stop_rule: SharedStoppingRuleV63,
    *,
    score_key: str = RAW_NLI_SCORE_KEY,
    acec_accept_key: str = ACEC_ACCEPT_SCORE_KEY,
    gold_cardinality: Optional[int] = None,
) -> EvidenceSelectionV63:
    ordered = trace.ordered_candidates()
    documents: Dict[str, List[EvidenceUnitV63]] = {}
    for unit in ordered:
        documents.setdefault(unit.document_id, []).append(unit)
    missing_documents = [
        document_id
        for document_id, units in documents.items()
        if not all(acec_accept_key in unit.scores for unit in units)
    ]
    if missing_documents:
        raise ValueError(
            "fixed trace lacks ACEC document acceptance replay for documents: "
            + ", ".join(missing_documents[:5])
        )
    accepted_documents = set()
    for document_id, units in documents.items():
        decisions = {
            float(unit.scores[acec_accept_key]) >= 0.5 for unit in units
        }
        if len(decisions) != 1:
            raise ValueError(
                f"ACEC acceptance must be document-level, not sentence-level: {document_id}"
            )
        if decisions == {True}:
            accepted_documents.add(document_id)
    eligible = [
        unit
        for unit in ordered
        if unit.document_id in accepted_documents
        and unit.official_hotpot_pair is not None
    ]
    rejected: Dict[str, str] = {}
    for unit in ordered:
        if unit.official_hotpot_pair is None:
            rejected[unit.evidence_id] = "excluded_unmappable_provenance"
        elif unit.document_id not in accepted_documents:
            rejected[unit.evidence_id] = "filtered_by_existing_acec_document_belief"
    return _selection(
        trace,
        eligible,
        selector_name="acec_existing_document_filter_plus_raw_nli",
        stop_rule=stop_rule,
        score_key=score_key,
        gold_cardinality=gold_cardinality,
        rejected_reasons=rejected,
    )


__all__ = [
    "ACEC_ACCEPT_SCORE_KEY",
    "ANSWER_HYPOTHESIS_VERSION",
    "RAW_NLI_SCORE_KEY",
    "SELECTOR_VERSION",
    "SharedStoppingRuleV63",
    "answer_hypothesis",
    "select_acec_existing_v63",
    "select_generic_nli_v63",
]
