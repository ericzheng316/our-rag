"""Immutable contracts for ACEC v7 learnable set selection.

V7 keeps the wide retrieval pool separate from the documents that are allowed
to update the live belief and enter the language-model context.  These
contracts deliberately contain no Hotpot supporting-fact labels: gold
supporting facts are evaluator/oracle data, never selector inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


V7_SCHEMA = "acec_belief_conditioned_set_selection"
V7_VERSION = 70
DEFAULT_CANDIDATE_POOL_SIZE = 20
DEFAULT_SELECTED_K = 5

_FORBIDDEN_SELECTOR_KEYS = {
    "supporting_facts",
    "supporting_fact",
    "sf_titles",
    "sf_title",
    "gold_sentence_ids",
    "gold_sent_ids",
    "gold_document_ids",
    "is_gold_sf",
    "gold_sf",
    "supporting_fact_membership_used",
}


def stable_id_hash(values: Sequence[str]) -> str:
    canonical = "\n".join(sorted({str(value) for value in values}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_sf_label_free(payload: Any, *, path: str = "root") -> None:
    """Fail closed if selector features accidentally contain SF supervision."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_SELECTOR_KEYS:
                raise ValueError(
                    f"selector payload contains forbidden SF label at {path}.{key}"
                )
            assert_sf_label_free(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_sf_label_free(value, path=f"{path}[{index}]")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class SelectorStateV7:
    question_id: str
    turn_index: int
    coverage: float
    coverage_std: float
    k_entropy: float
    slot_probabilities: Tuple[float, ...]
    slot_weights: Tuple[float, ...]
    slot_bound: Tuple[bool, ...]
    target_slot: Optional[int]
    retrieval_budget_remaining: int
    max_turns: int = 5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.question_id).strip():
            raise ValueError("selector state requires a question id")
        if self.turn_index < 0 or self.max_turns <= 0:
            raise ValueError("invalid selector turn index/max_turns")
        if self.retrieval_budget_remaining < 0:
            raise ValueError("retrieval budget must be non-negative")
        _probability(self.coverage, "coverage")
        if _finite(self.coverage_std, "coverage_std") < 0.0:
            raise ValueError("coverage_std must be non-negative")
        if _finite(self.k_entropy, "k_entropy") < 0.0:
            raise ValueError("k_entropy must be non-negative")
        size = len(self.slot_probabilities)
        if not size:
            raise ValueError("selector state requires at least one slot")
        if len(self.slot_weights) != size or len(self.slot_bound) != size:
            raise ValueError("slot probabilities/weights/bound flags must align")
        for index, value in enumerate(self.slot_probabilities):
            _probability(value, f"slot_probabilities[{index}]")
        for index, value in enumerate(self.slot_weights):
            if _finite(value, f"slot_weights[{index}]") < 0.0:
                raise ValueError("slot weights must be non-negative")
        if sum(self.slot_weights) <= 0.0:
            raise ValueError("slot weights must have positive mass")
        if self.target_slot is not None and not 0 <= self.target_slot < size:
            raise ValueError("target slot is outside the state")
        assert_sf_label_free(self.metadata, path="state.metadata")

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectorStateV7":
        return cls(
            question_id=str(payload["question_id"]),
            turn_index=int(payload["turn_index"]),
            coverage=float(payload["coverage"]),
            coverage_std=float(payload["coverage_std"]),
            k_entropy=float(payload["k_entropy"]),
            slot_probabilities=tuple(
                float(value) for value in payload["slot_probabilities"]
            ),
            slot_weights=tuple(float(value) for value in payload["slot_weights"]),
            slot_bound=tuple(bool(value) for value in payload["slot_bound"]),
            target_slot=(
                None
                if payload.get("target_slot") is None
                else int(payload["target_slot"])
            ),
            retrieval_budget_remaining=int(payload["retrieval_budget_remaining"]),
            max_turns=int(payload.get("max_turns", 5)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CandidateV7:
    candidate_id: str
    contents: str
    retrieval_rank: int
    retrieval_score: float
    slot_entailment: Tuple[float, ...]
    slot_hit_probabilities: Tuple[float, ...]
    slot_contradiction: Tuple[float, ...] = ()
    slot_neutral: Tuple[float, ...] = ()
    signed_answer_value: float = 0.0
    source_quality: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip() or not str(self.contents).strip():
            raise ValueError("candidate requires id and non-empty contents")
        if self.retrieval_rank < 0:
            raise ValueError("retrieval rank must be non-negative")
        _finite(self.retrieval_score, "retrieval_score")
        size = len(self.slot_entailment)
        if not size or len(self.slot_hit_probabilities) != size:
            raise ValueError("candidate slot entailment/hit vectors must align")
        for name, values in (
            ("slot_entailment", self.slot_entailment),
            ("slot_hit_probabilities", self.slot_hit_probabilities),
        ):
            for index, value in enumerate(values):
                _probability(value, f"{name}[{index}]")
        for name, values in (
            ("slot_contradiction", self.slot_contradiction),
            ("slot_neutral", self.slot_neutral),
        ):
            if values and len(values) != size:
                raise ValueError(f"{name} must be empty or align with slots")
            for index, value in enumerate(values):
                _probability(value, f"{name}[{index}]")
        _finite(self.signed_answer_value, "signed_answer_value")
        _probability(self.source_quality, "source_quality")
        assert_sf_label_free(self.metadata, path="candidate.metadata")

    @property
    def document_length(self) -> int:
        return len(self.contents.split())

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateV7":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            contents=str(payload["contents"]),
            retrieval_rank=int(payload["retrieval_rank"]),
            retrieval_score=float(payload.get("retrieval_score", 0.0)),
            slot_entailment=tuple(float(v) for v in payload["slot_entailment"]),
            slot_hit_probabilities=tuple(
                float(v) for v in payload["slot_hit_probabilities"]
            ),
            slot_contradiction=tuple(
                float(v) for v in payload.get("slot_contradiction", ())
            ),
            slot_neutral=tuple(float(v) for v in payload.get("slot_neutral", ())),
            signed_answer_value=float(payload.get("signed_answer_value", 0.0)),
            source_quality=float(payload.get("source_quality", 0.5)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SelectionStepV7:
    position: int
    selected_id: str
    log_probability: float
    score: float
    conditional_coverage_gain: float
    information_gain: float
    signed_answer_value: float
    candidate_probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.position < 0 or not self.selected_id:
            raise ValueError("invalid selection step")
        for name in (
            "log_probability",
            "score",
            "conditional_coverage_gain",
            "information_gain",
            "signed_answer_value",
        ):
            _finite(getattr(self, name), name)
        if self.selected_id not in self.candidate_probabilities:
            raise ValueError("selected id is missing from candidate probabilities")
        total = 0.0
        for candidate_id, value in self.candidate_probabilities.items():
            if not candidate_id:
                raise ValueError("empty candidate id in probability map")
            total += _probability(value, f"probability[{candidate_id}]")
        if abs(total - 1.0) > 1e-5:
            raise ValueError("candidate probabilities must sum to one")

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "candidate_probabilities": dict(self.candidate_probabilities)}


@dataclass(frozen=True)
class SelectionTraceV7:
    question_id: str
    turn_index: int
    strategy: str
    pool_ids: Tuple[str, ...]
    selected_ids: Tuple[str, ...]
    steps: Tuple[SelectionStepV7, ...]
    candidate_pool_size: int
    selected_k: int
    selector_artifact_sha256: str
    answer_value_artifact_sha256: str
    state_fingerprint_before: str
    schema: str = V7_SCHEMA
    version: int = V7_VERSION

    def __post_init__(self) -> None:
        if len(set(self.pool_ids)) != len(self.pool_ids):
            raise ValueError("candidate pool ids must be unique")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected ids must be unique")
        if not set(self.selected_ids).issubset(self.pool_ids):
            raise ValueError("selection contains ids outside the pool")
        if len(self.selected_ids) != len(self.steps):
            raise ValueError("selection steps must align with selected ids")
        if len(self.pool_ids) > self.candidate_pool_size:
            raise ValueError("pool exceeds declared candidate pool size")
        if len(self.selected_ids) > self.selected_k:
            raise ValueError("selection exceeds declared K")
        if tuple(step.selected_id for step in self.steps) != self.selected_ids:
            raise ValueError("selection order does not match steps")
        if not self.selector_artifact_sha256 or not self.answer_value_artifact_sha256:
            raise ValueError("selection trace requires frozen artifact identities")

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "steps": [step.to_dict() for step in self.steps],
            "unselected_ids": [
                candidate_id
                for candidate_id in self.pool_ids
                if candidate_id not in set(self.selected_ids)
            ],
        }


def state_fingerprint_v7(state: SelectorStateV7) -> str:
    payload = json.dumps(
        state.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CandidateV7",
    "DEFAULT_CANDIDATE_POOL_SIZE",
    "DEFAULT_SELECTED_K",
    "SelectionStepV7",
    "SelectionTraceV7",
    "SelectorStateV7",
    "V7_SCHEMA",
    "V7_VERSION",
    "assert_sf_label_free",
    "stable_id_hash",
    "state_fingerprint_v7",
]
