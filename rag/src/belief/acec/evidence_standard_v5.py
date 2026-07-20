"""Dataset-agnostic supervision contract for ACEC calibration v5.

The contract deliberately separates three concepts that older calibration
builders conflated:

* a *requirement* is one minimal piece of evidence needed by the task;
* an *assessment* states whether the exact document selected at runtime
  satisfies that requirement, together with its provenance and confidence;
* *fit eligibility* is a policy decision made after the assessment.  An
  ambiguous assessment is excluded instead of being forced into a negative
  class.

Dataset adapters may derive requirements from supporting sentences, citations,
human rationales, database rows, or another auditable source.  Source titles
and dataset-specific ids are provenance only; they never determine the target.
At inference time none of these annotations are used.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


STANDARD_NAME = "minimal_evidence_coverage"
STANDARD_VERSION = 1


class AnnotationScope(str, Enum):
    """What can be concluded when an annotated evidence span is absent."""

    CLOSED_CANDIDATE_POOL = "closed_candidate_pool"
    SUPPORT_ONLY = "support_only"


class AssessmentBasis(str, Enum):
    EXACT_SPAN = "exact_span"
    LEXICAL_SPAN = "lexical_span"
    CLOSED_WORLD_ABSENCE = "closed_world_absence"
    AMBIGUOUS_SPAN = "ambiguous_span"
    UNASSIGNED_REQUIREMENT = "unassigned_requirement"
    MISSING_ANNOTATION = "missing_annotation"


class UtilityBasis(str, Enum):
    """Interchangeable ways to instantiate the same marginal-utility target."""

    REQUIREMENT_COVERAGE = "requirement_coverage"
    ANSWER_SCORE = "answer_score"
    FROZEN_JUDGE = "frozen_judge"
    HUMAN_AUDIT = "human_audit"


@dataclass(frozen=True)
class EvidenceUnit:
    """One auditable span that can satisfy an evidence requirement."""

    unit_id: str
    text: str
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("evidence unit_id must be non-empty")
        if not normalize_text(self.text):
            raise ValueError(f"evidence unit {self.unit_id!r} has no text")


@dataclass(frozen=True)
class EvidenceRequirement:
    """A minimal semantic requirement, possibly supported by several spans."""

    requirement_id: str
    description: str
    evidence_units: Tuple[EvidenceUnit, ...]
    satisfaction_rule: str = "all"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.requirement_id:
            raise ValueError("evidence requirement_id must be non-empty")
        if not self.evidence_units:
            raise ValueError(
                f"evidence requirement {self.requirement_id!r} has no evidence units"
            )
        if self.satisfaction_rule not in {"all", "any"}:
            raise ValueError("evidence satisfaction_rule must be 'all' or 'any'")

    @property
    def matching_text(self) -> str:
        if normalize_text(self.description):
            return self.description
        return " ".join(unit.text for unit in self.evidence_units)


@dataclass(frozen=True)
class EvidenceSpecification:
    """All requirements and annotation semantics for one question."""

    question_id: str
    requirements: Tuple[EvidenceRequirement, ...]
    annotation_scope: AnnotationScope
    adapter_name: str
    annotation_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [requirement.requirement_id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence requirement ids must be unique within a question")


@runtime_checkable
class EvidenceSpecificationAdapter(Protocol):
    """Dataset boundary for the v5 calibration contract."""

    name: str

    def evidence_specification(self, record: Dict[str, Any]) -> EvidenceSpecification:
        ...


@dataclass(frozen=True)
class RequirementAssignment:
    slot_index: int
    requirement_index: Optional[int]
    similarity: Optional[float]
    margin: Optional[float]
    assessable: bool
    reason: str


@dataclass(frozen=True)
class CoverageAssessment:
    """A provenance-bearing target, not a dataset-specific class label."""

    coverage_value: Optional[float]
    confidence: float
    assessable: bool
    basis: AssessmentBasis
    requirement_id: Optional[str]
    evidence_unit_id: Optional[str]
    match_score: Optional[float]
    rationale: str

    def binary_target(self, min_confidence: float = 0.90) -> Optional[bool]:
        """Return a high-confidence endpoint for the current binary calibrator.

        The v5 audit retains the continuous value and confidence.  The current
        observation model is Bernoulli, so only confident 0/1 endpoints enter
        fitting; intermediate or unassessable cases stay visible in the audit.
        """

        if not self.assessable or self.coverage_value is None:
            return None
        if self.confidence < min_confidence:
            return None
        if self.coverage_value == 1.0:
            return True
        if self.coverage_value == 0.0:
            return False
        return None


@dataclass(frozen=True)
class MarginalUtilityAssessment:
    """Canonical v5 target: utility gained by adding one selected document.

    ``utility_before`` and ``utility_after`` are measured under the same
    evaluator.  The delta is continuous and can be zero for a relevant but
    redundant document.  ``new_requirement_coverage`` is the Bernoulli event
    consumed by the current ACEC filter when requirement annotations are the
    evaluator; it is not a positive/negative/unknown dataset label.
    """

    utility_before: Optional[float]
    utility_after: Optional[float]
    marginal_utility: Optional[float]
    confidence: float
    assessable: bool
    basis: UtilityBasis
    requirement_id: Optional[str]
    evidence_unit_id: Optional[str]
    static_support: Optional[float]
    new_requirement_coverage: Optional[bool]
    repeated_document: bool
    rationale: str

    def binary_gain_target(self, min_confidence: float = 0.90) -> Optional[bool]:
        if not self.assessable or self.new_requirement_coverage is None:
            return None
        if self.confidence < min_confidence:
            return None
        return bool(self.new_requirement_coverage)


@dataclass
class RequirementCoverageHistory:
    """History needed to turn static support into conditional marginal gain."""

    requirement_count: int
    covered_requirement_ids: set[str] = field(default_factory=set)
    selected_document_fingerprints: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.requirement_count <= 0:
            raise ValueError("requirement_count must be positive")

    @property
    def utility(self) -> float:
        return len(self.covered_requirement_ids) / self.requirement_count

    def observe(
        self,
        assessment: CoverageAssessment,
        selected_document_text: str,
        document_seen_before: Optional[bool] = None,
        record_document: bool = True,
        covered_requirement_ids_before: Optional[set[str]] = None,
        commit_coverage: bool = True,
    ) -> MarginalUtilityAssessment:
        """Apply one assessment in trajectory order and return its delta."""

        fingerprint = normalize_text(selected_document_text)
        repeated_document = (
            bool(fingerprint and fingerprint in self.selected_document_fingerprints)
            if document_seen_before is None
            else bool(document_seen_before)
        )
        if fingerprint and record_document:
            self.selected_document_fingerprints.add(fingerprint)

        covered_before = (
            set(self.covered_requirement_ids)
            if covered_requirement_ids_before is None
            else set(covered_requirement_ids_before)
        )
        before = len(covered_before) / self.requirement_count
        if (
            not assessment.assessable
            or assessment.coverage_value is None
            or assessment.requirement_id is None
        ):
            return MarginalUtilityAssessment(
                utility_before=before,
                utility_after=None,
                marginal_utility=None,
                confidence=assessment.confidence,
                assessable=False,
                basis=UtilityBasis.REQUIREMENT_COVERAGE,
                requirement_id=assessment.requirement_id,
                evidence_unit_id=assessment.evidence_unit_id,
                static_support=assessment.coverage_value,
                new_requirement_coverage=None,
                repeated_document=repeated_document,
                rationale="static evidence assessment is not auditable enough for a delta",
            )

        newly_covered = bool(
            assessment.coverage_value == 1.0
            and assessment.requirement_id not in covered_before
        )
        if newly_covered and commit_coverage:
            self.covered_requirement_ids.add(assessment.requirement_id)
        after = before + (1.0 / self.requirement_count if newly_covered else 0.0)
        return MarginalUtilityAssessment(
            utility_before=before,
            utility_after=after,
            marginal_utility=after - before,
            confidence=assessment.confidence,
            assessable=True,
            basis=UtilityBasis.REQUIREMENT_COVERAGE,
            requirement_id=assessment.requirement_id,
            evidence_unit_id=assessment.evidence_unit_id,
            static_support=assessment.coverage_value,
            new_requirement_coverage=newly_covered,
            repeated_document=repeated_document,
            rationale=(
                "selected document covers a previously unsatisfied requirement"
                if newly_covered
                else "selected document adds no new requirement coverage"
            ),
        )


def marginal_utility_from_scores(
    score_before: Optional[float],
    score_after: Optional[float],
    basis: UtilityBasis,
    confidence: float,
    rationale: str,
) -> MarginalUtilityAssessment:
    """Generic constructor for answer-score, frozen-judge, or human utility.

    This is the path for datasets without supporting-evidence annotations.  A
    caller can use EM/F1, answer log-probability, a frozen answerability judge,
    or human scoring, provided the same evaluator is used before and after the
    new evidence is added.
    """

    assessable = score_before is not None and score_after is not None
    delta = float(score_after - score_before) if assessable else None
    return MarginalUtilityAssessment(
        utility_before=score_before,
        utility_after=score_after,
        marginal_utility=delta,
        confidence=float(confidence),
        assessable=assessable,
        basis=basis,
        requirement_id=None,
        evidence_unit_id=None,
        static_support=None,
        new_requirement_coverage=None,
        repeated_document=False,
        rationale=rationale,
    )


def standard_payload() -> Dict[str, Any]:
    """Stable artifact declaration for loader compatibility checks."""

    return {
        "name": STANDARD_NAME,
        "version": STANDARD_VERSION,
        "requirement": "minimal evidence needed to resolve one latent slot",
        "target": "conditional marginal utility of the selected runtime document",
        "definition": "U(E_before union {d_t}, q) - U(E_before, q)",
        "supported_bases": [basis.value for basis in UtilityBasis],
        "static_support_is_intermediate_only": True,
        "source_ids_are_targets": False,
        "ambiguous_examples_are_fit_targets": False,
    }


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _tokens(value: str) -> List[str]:
    normalized = normalize_text(value)
    return normalized.split() if normalized else []


def _token_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    common = sum((Counter(left) & Counter(right)).values())
    if common == 0:
        return 0.0
    precision = common / len(right)
    recall = common / len(left)
    return 2.0 * precision * recall / (precision + recall)


def document_novelty_score(
    selected_document_text: str,
    prior_document_texts: Sequence[str],
) -> float:
    """Cheap, deterministic novelty feature shared by offline and runtime.

    It is one minus the maximum token overlap with prior selected documents.
    Exact repeats therefore score zero while a genuinely new passage tends
    toward one.  This is intentionally independent of the NLI support score.
    """

    selected_tokens = _tokens(selected_document_text)
    if not selected_tokens:
        return 0.0
    if not prior_document_texts:
        return 1.0
    maximum_overlap = max(
        (_token_f1(selected_tokens, _tokens(prior)) for prior in prior_document_texts),
        default=0.0,
    )
    return float(min(max(1.0 - maximum_overlap, 0.0), 1.0))


def evidence_span_match(unit_text: str, document_text: str) -> Tuple[float, AssessmentBasis]:
    """Measure whether an annotated evidence span occurs in a selected doc.

    Exact normalized containment is the high-precision path.  The lexical
    fallback compares the evidence unit with sentence-sized windows so minor
    corpus formatting differences do not turn a real span into a false miss.
    It does not use titles or the ACEC NLI score being calibrated.
    """

    unit_normalized = normalize_text(unit_text)
    document_normalized = normalize_text(document_text)
    if not unit_normalized or not document_normalized:
        return 0.0, AssessmentBasis.LEXICAL_SPAN
    if unit_normalized in document_normalized:
        return 1.0, AssessmentBasis.EXACT_SPAN

    unit_tokens = _tokens(unit_text)
    sentence_candidates = [
        segment for segment in re.split(r"(?<=[.!?])\s+|\n+", str(document_text)) if segment.strip()
    ]
    if not sentence_candidates:
        sentence_candidates = [str(document_text)]
    score = max(
        (_token_f1(unit_tokens, _tokens(candidate)) for candidate in sentence_candidates),
        default=0.0,
    )
    return float(score), AssessmentBasis.LEXICAL_SPAN


def assess_selected_document(
    requirement: Optional[EvidenceRequirement],
    selected_document_text: str,
    annotation_scope: AnnotationScope,
    support_threshold: float = 0.90,
    absence_threshold: float = 0.35,
) -> CoverageAssessment:
    """Assess an exact runtime-selected document against one requirement."""

    if requirement is None:
        return CoverageAssessment(
            coverage_value=None,
            confidence=0.0,
            assessable=False,
            basis=AssessmentBasis.UNASSIGNED_REQUIREMENT,
            requirement_id=None,
            evidence_unit_id=None,
            match_score=None,
            rationale="slot could not be assigned to one evidence requirement",
        )
    if not requirement.evidence_units:
        return CoverageAssessment(
            coverage_value=None,
            confidence=0.0,
            assessable=False,
            basis=AssessmentBasis.MISSING_ANNOTATION,
            requirement_id=requirement.requirement_id,
            evidence_unit_id=None,
            match_score=None,
            rationale="assigned requirement has no auditable evidence span",
        )

    candidates = []
    for unit in requirement.evidence_units:
        score, basis = evidence_span_match(unit.text, selected_document_text)
        candidates.append((score, basis, unit))
    best_score, best_basis, best_unit = max(candidates, key=lambda item: item[0])
    required_score = (
        min(score for score, _, _ in candidates)
        if requirement.satisfaction_rule == "all"
        else best_score
    )
    matched_unit_id = (
        ",".join(unit.unit_id for _, _, unit in candidates)
        if requirement.satisfaction_rule == "all"
        else best_unit.unit_id
    )

    if required_score >= support_threshold:
        return CoverageAssessment(
            coverage_value=1.0,
            confidence=float(required_score),
            assessable=True,
            basis=best_basis,
            requirement_id=requirement.requirement_id,
            evidence_unit_id=matched_unit_id,
            match_score=float(required_score),
            rationale=(
                "selected document satisfies the requirement's annotated evidence spans"
            ),
        )

    if (
        annotation_scope == AnnotationScope.CLOSED_CANDIDATE_POOL
        and best_score <= absence_threshold
    ):
        return CoverageAssessment(
            coverage_value=0.0,
            confidence=float(1.0 - best_score),
            assessable=True,
            basis=AssessmentBasis.CLOSED_WORLD_ABSENCE,
            requirement_id=requirement.requirement_id,
            evidence_unit_id=matched_unit_id,
            match_score=float(best_score),
            rationale=(
                "closed candidate-pool annotation and no assigned evidence span "
                "is present in the selected document"
            ),
        )

    return CoverageAssessment(
        coverage_value=None,
        confidence=float(max(best_score, 1.0 - best_score)),
        assessable=False,
        basis=AssessmentBasis.AMBIGUOUS_SPAN,
        requirement_id=requirement.requirement_id,
        evidence_unit_id=matched_unit_id,
        match_score=float(required_score),
        rationale="span evidence is neither a confident match nor a justified absence",
    )


def assign_slots_to_requirements(
    slot_hypotheses: Sequence[str],
    requirements: Sequence[EvidenceRequirement],
    embedder: Any,
    min_similarity: float = 0.30,
    min_margin: float = 0.05,
) -> Dict[int, RequirementAssignment]:
    """One-to-one semantic assignment with explicit ambiguity abstention."""

    if not slot_hypotheses:
        return {}
    if not requirements:
        return {
            slot_index: RequirementAssignment(
                slot_index, None, None, None, False, "no_evidence_requirements"
            )
            for slot_index in range(len(slot_hypotheses))
        }

    texts = list(slot_hypotheses) + [requirement.matching_text for requirement in requirements]
    embeddings = np.asarray(
        embedder.encode(texts, show_progress_bar=False), dtype=np.float64
    )
    slot_embeddings = embeddings[: len(slot_hypotheses)]
    requirement_embeddings = embeddings[len(slot_hypotheses) :]
    slot_norms = np.linalg.norm(slot_embeddings, axis=1, keepdims=True) + 1e-12
    requirement_norms = np.linalg.norm(requirement_embeddings, axis=1, keepdims=True) + 1e-12
    similarities = (slot_embeddings / slot_norms) @ (
        requirement_embeddings / requirement_norms
    ).T

    candidates = []
    per_slot_margin: Dict[int, float] = {}
    for slot_index in range(len(slot_hypotheses)):
        ordered = np.sort(similarities[slot_index])[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        per_slot_margin[slot_index] = margin
        for requirement_index in range(len(requirements)):
            candidates.append(
                (float(similarities[slot_index, requirement_index]), slot_index, requirement_index)
            )

    assignments: Dict[int, RequirementAssignment] = {}
    used_requirements = set()
    for similarity, slot_index, requirement_index in sorted(candidates, reverse=True):
        if slot_index in assignments or requirement_index in used_requirements:
            continue
        margin = per_slot_margin[slot_index]
        if similarity < min_similarity:
            continue
        if len(requirements) > 1 and margin < min_margin:
            continue
        assignments[slot_index] = RequirementAssignment(
            slot_index=slot_index,
            requirement_index=requirement_index,
            similarity=similarity,
            margin=margin,
            assessable=True,
            reason="semantic_requirement_assignment",
        )
        used_requirements.add(requirement_index)

    for slot_index in range(len(slot_hypotheses)):
        if slot_index in assignments:
            continue
        best = float(np.max(similarities[slot_index]))
        margin = per_slot_margin[slot_index]
        reason = "assignment_similarity_below_threshold"
        if best >= min_similarity and len(requirements) > 1 and margin < min_margin:
            reason = "assignment_ambiguous"
        elif best >= min_similarity:
            reason = "requirement_already_claimed"
        assignments[slot_index] = RequirementAssignment(
            slot_index=slot_index,
            requirement_index=None,
            similarity=best,
            margin=margin,
            assessable=False,
            reason=reason,
        )
    return assignments


__all__ = [
    "AnnotationScope",
    "AssessmentBasis",
    "CoverageAssessment",
    "EvidenceRequirement",
    "EvidenceSpecification",
    "EvidenceSpecificationAdapter",
    "EvidenceUnit",
    "MarginalUtilityAssessment",
    "RequirementCoverageHistory",
    "RequirementAssignment",
    "STANDARD_NAME",
    "STANDARD_VERSION",
    "assess_selected_document",
    "assign_slots_to_requirements",
    "document_novelty_score",
    "evidence_span_match",
    "marginal_utility_from_scores",
    "normalize_text",
    "standard_payload",
    "UtilityBasis",
]
