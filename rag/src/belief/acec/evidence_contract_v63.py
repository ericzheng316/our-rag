"""Auditable, dataset-agnostic evidence contracts for ACEC v6.3.

The runtime trace deliberately contains no gold answer or supporting-fact
membership.  Benchmark adapters keep those labels in a separate evaluator
manifest.  This separation is enforced here rather than left to launcher
convention, because a fixed-trace selector comparison is invalid if the trace
itself leaks HotpotQA labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


EVIDENCE_SCHEMA = "acec_evidence_trace"
EVIDENCE_SCHEMA_VERSION = 63
PROVENANCE_STATUSES = frozenset({"native", "derived", "unmappable"})

# Exact normalized key names that are evaluator-side only.  ``draft_answer``
# is intentionally allowed; a generic ``answer`` key is not, so accidental
# reuse of a labeled HotpotQA record fails closed.
_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold_answer",
        "gold_answers",
        "golden_answer",
        "golden_answers",
        "gold_cardinality",
        "gold_sp",
        "gold_support",
        "gold_supporting_facts",
        "supporting_fact",
        "supporting_facts",
        "is_supporting_fact",
        "sf_label",
        "_annotation_record",
    }
)


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def assert_runtime_payload_has_no_gold(payload: Any, path: str = "$") -> None:
    """Recursively reject evaluator-only labels from a runtime payload."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"runtime payload leaks evaluator field {path}.{key}")
            assert_runtime_payload_has_no_gold(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_runtime_payload_has_no_gold(value, f"{path}[{index}]")


def _validated_scores(scores: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, raw_value in scores.items():
        name = str(key).strip()
        if not name:
            raise ValueError("evidence score names must be non-empty")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"evidence score {name!r} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evidence score {name!r} must be in [0, 1]")
        result[name] = value
    return result


@dataclass(frozen=True)
class EvidenceUnitV63:
    """One sentence-sized candidate with auditable source identity.

    ``retrieval_turn`` and ``retrieval_rank`` denote first observation.  A
    repeated retrieval is represented by another reference to the same
    ``evidence_id`` in a later :class:`EvidenceTurnV63`, not by duplicating the
    unit and inventing another official sentence identity.
    """

    evidence_id: str
    source_id: Optional[str]
    unit_index: Optional[int]
    text: str
    document_id: str
    retrieval_turn: int
    retrieval_rank: int
    provenance_status: str = "native"
    scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.evidence_id).strip():
            raise ValueError("evidence_id must be non-empty")
        if not str(self.document_id).strip():
            raise ValueError("document_id must be non-empty")
        if not str(self.text).strip():
            raise ValueError(f"evidence {self.evidence_id!r} has no text")
        if self.retrieval_turn < 0 or self.retrieval_rank < 0:
            raise ValueError("retrieval turn/rank must be non-negative")
        if self.provenance_status not in PROVENANCE_STATUSES:
            raise ValueError(
                f"unsupported provenance_status {self.provenance_status!r}"
            )
        if self.unit_index is not None and self.unit_index < 0:
            raise ValueError("unit_index must be non-negative when present")
        if self.provenance_status == "native":
            if not str(self.source_id or "").strip() or self.unit_index is None:
                raise ValueError("native evidence requires source_id and unit_index")
        object.__setattr__(self, "scores", _validated_scores(self.scores))
        assert_runtime_payload_has_no_gold(dict(self.metadata), "$.metadata")

    @property
    def official_hotpot_pair(self) -> Optional[Tuple[str, int]]:
        """Return an official pair only for native benchmark provenance."""

        if self.provenance_status != "native":
            return None
        assert self.source_id is not None and self.unit_index is not None
        return str(self.source_id), int(self.unit_index)

    def with_scores(self, updates: Mapping[str, float]) -> "EvidenceUnitV63":
        merged = dict(self.scores)
        merged.update(updates)
        return EvidenceUnitV63(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            unit_index=self.unit_index,
            text=self.text,
            document_id=self.document_id,
            retrieval_turn=self.retrieval_turn,
            retrieval_rank=self.retrieval_rank,
            provenance_status=self.provenance_status,
            scores=merged,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceUnitV63":
        assert_runtime_payload_has_no_gold(payload)
        return cls(
            evidence_id=str(payload["evidence_id"]),
            source_id=(
                None if payload.get("source_id") is None else str(payload["source_id"])
            ),
            unit_index=(
                None if payload.get("unit_index") is None else int(payload["unit_index"])
            ),
            text=str(payload["text"]),
            document_id=str(payload["document_id"]),
            retrieval_turn=int(payload["retrieval_turn"]),
            retrieval_rank=int(payload["retrieval_rank"]),
            provenance_status=str(payload.get("provenance_status", "native")),
            scores=dict(payload.get("scores") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceTurnV63:
    turn: int
    query: str
    evidence_ids: Tuple[str, ...]
    action_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if not str(self.query).strip():
            raise ValueError("fixed evidence turns require a non-empty query")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("one turn cannot contain duplicate evidence ids")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceTurnV63":
        assert_runtime_payload_has_no_gold(payload)
        return cls(
            turn=int(payload["turn"]),
            query=str(payload["query"]),
            evidence_ids=tuple(str(value) for value in payload.get("evidence_ids", [])),
            action_mode=(
                None if payload.get("action_mode") is None else str(payload["action_mode"])
            ),
        )


@dataclass(frozen=True)
class EvidenceTraceV63:
    """Fixed selector input: q, short answer, ordered turns, and candidates."""

    question_id: str
    question: str
    draft_answer: str
    candidates: Tuple[EvidenceUnitV63, ...]
    turns: Tuple[EvidenceTurnV63, ...]
    answer_contract_version: str
    schema: str = EVIDENCE_SCHEMA
    schema_version: int = EVIDENCE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_SCHEMA or self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evidence schema {self.schema!r} v{self.schema_version}"
            )
        if not str(self.question_id).strip() or not str(self.question).strip():
            raise ValueError("trace requires question_id and question")
        if not str(self.draft_answer).strip():
            raise ValueError("trace requires a parsed non-empty draft_answer")
        if not str(self.answer_contract_version).strip():
            raise ValueError("trace requires answer_contract_version")
        if not self.candidates or not self.turns:
            raise ValueError("fixed selector trace requires candidates and turns")
        candidate_ids = [unit.evidence_id for unit in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate evidence ids must be unique")
        turns = [turn.turn for turn in self.turns]
        if turns != sorted(turns) or len(turns) != len(set(turns)):
            raise ValueError("trace turns must be strictly ordered and unique")
        known = set(candidate_ids)
        referenced = {
            evidence_id for turn in self.turns for evidence_id in turn.evidence_ids
        }
        unknown = referenced - known
        if unknown:
            raise ValueError(f"trace turns reference unknown evidence ids: {sorted(unknown)}")
        if known != referenced:
            missing = sorted(known - referenced)
            raise ValueError(f"candidates absent from ordered trace: {missing}")
        assert_runtime_payload_has_no_gold(dict(self.metadata), "$.metadata")

    @property
    def candidate_by_id(self) -> Dict[str, EvidenceUnitV63]:
        return {unit.evidence_id: unit for unit in self.candidates}

    def ordered_candidates(self) -> Tuple[EvidenceUnitV63, ...]:
        by_id = self.candidate_by_id
        ordered = []
        seen = set()
        for turn in self.turns:
            for evidence_id in turn.evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    ordered.append(by_id[evidence_id])
        return tuple(ordered)

    def with_candidates(self, candidates: Sequence[EvidenceUnitV63]) -> "EvidenceTraceV63":
        return EvidenceTraceV63(
            question_id=self.question_id,
            question=self.question,
            draft_answer=self.draft_answer,
            candidates=tuple(candidates),
            turns=self.turns,
            answer_contract_version=self.answer_contract_version,
            schema=self.schema,
            schema_version=self.schema_version,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "question": self.question,
            "draft_answer": self.draft_answer,
            "answer_contract_version": self.answer_contract_version,
            "candidate_evidence": [unit.to_dict() for unit in self.candidates],
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceTraceV63":
        assert_runtime_payload_has_no_gold(payload)
        return cls(
            schema=str(payload.get("schema", "")),
            schema_version=int(payload.get("schema_version", -1)),
            question_id=str(payload["question_id"]),
            question=str(payload["question"]),
            draft_answer=str(payload["draft_answer"]),
            answer_contract_version=str(payload["answer_contract_version"]),
            candidates=tuple(
                EvidenceUnitV63.from_dict(value)
                for value in payload.get("candidate_evidence", [])
            ),
            turns=tuple(
                EvidenceTurnV63.from_dict(value) for value in payload.get("turns", [])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceSelectionV63:
    selector_name: str
    selector_version: str
    selected_evidence_ids: Tuple[str, ...]
    score_key: str
    stop_mode: str
    stop_value: float
    reasons: Mapping[str, str] = field(default_factory=dict)

    def validate_against(self, trace: EvidenceTraceV63) -> None:
        selected = list(self.selected_evidence_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("selected evidence ids must be unique")
        unknown = set(selected) - set(trace.candidate_by_id)
        if unknown:
            raise ValueError(f"selection references unknown evidence ids: {sorted(unknown)}")

    def official_hotpot_sp(self, trace: EvidenceTraceV63) -> Tuple[Tuple[str, int], ...]:
        self.validate_against(trace)
        by_id = trace.candidate_by_id
        pairs = {
            pair
            for evidence_id in self.selected_evidence_ids
            if (pair := by_id[evidence_id].official_hotpot_pair) is not None
        }
        return tuple(sorted(pairs, key=lambda value: (value[0], value[1])))


def native_hotpot_pairs(units: Iterable[EvidenceUnitV63]) -> set[Tuple[str, int]]:
    return {
        pair
        for unit in units
        if (pair := unit.official_hotpot_pair) is not None
    }


__all__ = [
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceSelectionV63",
    "EvidenceTraceV63",
    "EvidenceTurnV63",
    "EvidenceUnitV63",
    "assert_runtime_payload_has_no_gold",
    "native_hotpot_pairs",
]
