"""Inference-time ACEC controller contracts for v6.4.

This module is deliberately free of torch/vLLM dependencies.  It turns an
already-computed belief state into explicit actions:

* leave the policy query unchanged;
* target the lowest-coverage slot and inject a confirmed entity;
* reject a premature answer and retrieve for the unresolved slot;
* force an answer when the frozen, calibrated stop rule fires.

Gold bridge entities are accepted only through the explicit ``gold_entity``
ablation mode.  Deployable modes never read evaluator annotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CONTROLLER_SCHEMA = "acec_inference_controller"
CONTROLLER_VERSION = 64
BRIDGE_ENTITY_SCHEMA = "acec_bridge_entity_label"
SLOT_CONTRACT_VERSION = "question_slots_v64.1"
FORCE_ANSWER_INSTRUCTION = (
    "The evidence budget is complete. Do not issue another retrieval query. "
    "Return the shortest final answer using the required final-answer format."
)


class ControllerMode(str, Enum):
    NONE = "none"
    MONITOR_PRESEED = "monitor_preseed"
    BELIEF_GAP = "belief_gap"
    BELIEF_ADAPTIVE = "belief_adaptive"
    BELIEF_GAP_ADAPTIVE = "belief_gap_adaptive"
    GOLD_ENTITY = "gold_entity"
    GOLD_ENTITY_ADAPTIVE = "gold_entity_adaptive"


class ControllerAction(str, Enum):
    ACCEPT_POLICY_ANSWER = "accept_policy_answer"
    EXECUTE_QUERY = "execute_query"
    OVERRIDE_ANSWER_WITH_QUERY = "override_answer_with_query"
    FORCE_ANSWER = "force_answer"


@dataclass(frozen=True)
class SlotDecompositionV64:
    question_id: str
    question: str
    slots: Tuple[str, ...]
    contract_version: str = SLOT_CONTRACT_VERSION
    generator: str = "unknown"

    def __post_init__(self) -> None:
        if not self.question_id or not self.question.strip():
            raise ValueError("slot decomposition requires id and question")
        cleaned = tuple(" ".join(slot.split()) for slot in self.slots)
        if not cleaned or any(not slot for slot in cleaned):
            raise ValueError("slot decomposition must contain non-empty slots")
        if len({slot.casefold() for slot in cleaned}) != len(cleaned):
            raise ValueError("slot decomposition contains duplicate requirements")
        object.__setattr__(self, "slots", cleaned)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SlotDecompositionV64":
        return cls(
            question_id=str(payload["question_id"]),
            question=str(payload["question"]),
            slots=tuple(str(value) for value in payload["slots"]),
            contract_version=str(payload.get("contract_version", "")),
            generator=str(payload.get("generator", "unknown")),
        )


_SLOT_TAG = re.compile(r"<slot>\s*(.*?)\s*</slot>", re.IGNORECASE | re.DOTALL)


def slot_decomposition_prompt(question: str, expected_k: int) -> str:
    if expected_k <= 0:
        raise ValueError("expected_k must be positive")
    return (
        f"Question: {question}\n"
        f"List exactly {expected_k} minimal evidence requirements needed to answer. "
        "Do not answer the question. Output only one <slot>...</slot> element "
        "per requirement, in dependency order."
    )


def parse_slot_decomposition(
    text: str,
    *,
    question_id: str,
    question: str,
    expected_k: int,
    generator: str,
) -> SlotDecompositionV64:
    matches = [" ".join(value.split()) for value in _SLOT_TAG.findall(str(text))]
    if len(matches) != expected_k:
        raise ValueError(
            f"slot parser expected {expected_k} tags but found {len(matches)}"
        )
    return SlotDecompositionV64(
        question_id=question_id,
        question=question,
        slots=tuple(matches),
        generator=generator,
    )


def preseed_belief_slots(
    belief: Any, slots: Sequence[str], *, freeze_slot_count: bool = True
) -> None:
    """Seed question-derived slots into a freshly reset ACEC runtime."""

    coverage = belief.coverage_belief
    if len(coverage.slots) != 0 or coverage.turn != 0:
        raise ValueError("slot preseed requires a freshly reset belief")
    if len(slots) > int(coverage.config.k_max):
        raise ValueError(
            f"slot count {len(slots)} exceeds belief k_max={coverage.config.k_max}"
        )
    for hypothesis in slots:
        coverage.spawn_slot(str(hypothesis))
    if freeze_slot_count:
        # Once a full question-derived requirement set is present, a low E5
        # similarity should route to the closest known slot rather than spawn
        # a third Hotpot slot and invalidate a fixed-K posterior.  Setting the
        # routing floor below cosine's range preserves REWRITE/EXPAND routing
        # while making the preseed count structural.
        belief.labeler.tau_new = -1.0
        belief.v64_slot_count_frozen = True


@dataclass(frozen=True)
class BeliefSnapshotV64:
    coverage: float
    coverage_std: float
    stop_voi: bool
    target_slot: Optional[int]
    target_hypothesis: Optional[str]
    target_probability: Optional[float]
    bound_entities: Tuple[str, ...]
    slot_probabilities: Tuple[float, ...]
    slot_bound: Tuple[bool, ...]

    @classmethod
    def from_belief(cls, belief: Any) -> "BeliefSnapshotV64":
        coverage = belief.coverage_belief
        target = coverage.suggest_target_slot()
        slots = list(coverage.slots)
        entities = tuple(
            dict.fromkeys(
                str(entity)
                for slot in slots
                if slot.bound
                for entity in slot.bound_entities
                if str(entity).strip()
            )
        )
        return cls(
            coverage=float(coverage.coverage()),
            coverage_std=float(math.sqrt(max(coverage.coverage_variance(), 0.0))),
            stop_voi=bool(coverage.should_stop_voi()),
            target_slot=target,
            target_hypothesis=(
                str(slots[target].hypothesis) if target is not None else None
            ),
            target_probability=(
                float(coverage.p[target]) if target is not None else None
            ),
            bound_entities=entities,
            slot_probabilities=tuple(float(value) for value in coverage.p),
            slot_bound=tuple(bool(slot.bound) for slot in slots),
        )


@dataclass(frozen=True)
class ControllerConfigV64:
    mode: ControllerMode = ControllerMode.NONE
    min_retrievals: int = 1
    max_retrievals: int = 5
    injection_after_retrievals: int = 1
    stop_coverage_min: float = 0.75
    stop_std_max: float = 0.25
    query_strategy: str = "slot_plus_entity"
    block_low_coverage_answer: bool = True

    def __post_init__(self) -> None:
        if self.min_retrievals < 0:
            raise ValueError("min_retrievals must be non-negative")
        if self.max_retrievals < max(self.min_retrievals, 1):
            raise ValueError("max_retrievals must be >= max(min_retrievals, 1)")
        if self.injection_after_retrievals < 0:
            raise ValueError("injection_after_retrievals must be non-negative")
        if not 0.0 <= self.stop_coverage_min <= 1.0:
            raise ValueError("stop_coverage_min must be in [0, 1]")
        if not 0.0 <= self.stop_std_max <= 1.0:
            raise ValueError("stop_std_max must be in [0, 1]")
        if self.query_strategy not in {
            "append_entity",
            "slot_only",
            "slot_plus_entity",
        }:
            raise ValueError(f"unsupported query_strategy {self.query_strategy!r}")

    @property
    def uses_gap(self) -> bool:
        return self.mode in {
            ControllerMode.BELIEF_GAP,
            ControllerMode.BELIEF_GAP_ADAPTIVE,
        }

    @property
    def uses_gold_entity(self) -> bool:
        return self.mode in {
            ControllerMode.GOLD_ENTITY,
            ControllerMode.GOLD_ENTITY_ADAPTIVE,
        }

    @property
    def uses_adaptive_stop(self) -> bool:
        return self.mode in {
            ControllerMode.BELIEF_ADAPTIVE,
            ControllerMode.BELIEF_GAP_ADAPTIVE,
            ControllerMode.GOLD_ENTITY_ADAPTIVE,
        }


@dataclass(frozen=True)
class ControllerDecisionV64:
    action: ControllerAction
    query: Optional[str]
    reason: str
    injected_entity: Optional[str]
    target_slot: Optional[int]
    snapshot: Optional[BeliefSnapshotV64]


def _normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def build_gap_directed_query(
    policy_query: str,
    target_hypothesis: Optional[str],
    entity: Optional[str],
    *,
    strategy: str,
) -> str:
    base = _normalize_space(policy_query)
    target = _normalize_space(target_hypothesis or "")
    bound_entity = _normalize_space(entity or "")
    if strategy == "append_entity":
        pieces = [base, bound_entity]
    elif strategy == "slot_only":
        pieces = [target or base]
    elif strategy == "slot_plus_entity":
        pieces = [target or base, bound_entity]
    else:
        raise ValueError(f"unsupported query strategy {strategy!r}")

    result = []
    seen = set()
    for piece in pieces:
        key = piece.casefold()
        if piece and key not in seen:
            seen.add(key)
            result.append(piece)
    if not result:
        raise ValueError("controller could not construct a retrieval query")
    return " ".join(result)


class BeliefInferenceControllerV64:
    def __init__(self, config: ControllerConfigV64):
        self.config = config

    def _ready_to_stop(
        self, snapshot: BeliefSnapshotV64, retrieval_calls: int
    ) -> bool:
        return bool(
            retrieval_calls >= self.config.min_retrievals
            and snapshot.stop_voi
            and snapshot.coverage >= self.config.stop_coverage_min
            and snapshot.coverage_std <= self.config.stop_std_max
        )

    def decide(
        self,
        *,
        question: str,
        belief: Optional[Any],
        retrieval_calls: int,
        policy_query: Optional[str] = None,
        policy_answer: Optional[str] = None,
        gold_entity: Optional[str] = None,
    ) -> ControllerDecisionV64:
        if bool(policy_query) == bool(policy_answer):
            raise ValueError("controller requires exactly one policy query or answer")
        snapshot = BeliefSnapshotV64.from_belief(belief) if belief is not None else None

        if self.config.mode in {
            ControllerMode.NONE,
            ControllerMode.MONITOR_PRESEED,
        }:
            return ControllerDecisionV64(
                action=(
                    ControllerAction.ACCEPT_POLICY_ANSWER
                    if policy_answer
                    else ControllerAction.EXECUTE_QUERY
                ),
                query=policy_query,
                reason=(
                    "controller_disabled"
                    if self.config.mode == ControllerMode.NONE
                    else "monitor_only"
                ),
                injected_entity=None,
                target_slot=None,
                snapshot=snapshot,
            )
        if snapshot is None:
            raise ValueError(f"controller mode {self.config.mode.value} requires belief")

        if retrieval_calls >= self.config.max_retrievals:
            if policy_answer:
                return ControllerDecisionV64(
                    ControllerAction.ACCEPT_POLICY_ANSWER,
                    None,
                    "maximum_retrieval_budget_reached",
                    None,
                    snapshot.target_slot,
                    snapshot,
                )
            return ControllerDecisionV64(
                ControllerAction.FORCE_ANSWER,
                None,
                "maximum_retrieval_budget_reached",
                None,
                snapshot.target_slot,
                snapshot,
            )

        ready_to_stop = (
            self.config.uses_adaptive_stop
            and self._ready_to_stop(snapshot, retrieval_calls)
        )
        if policy_answer and (
            not self.config.uses_adaptive_stop
            or ready_to_stop
            or not self.config.block_low_coverage_answer
        ):
            return ControllerDecisionV64(
                ControllerAction.ACCEPT_POLICY_ANSWER,
                None,
                "policy_answer_accepted"
                if not ready_to_stop
                else "belief_stop_accepts_answer",
                None,
                snapshot.target_slot,
                snapshot,
            )
        if policy_query and ready_to_stop:
            return ControllerDecisionV64(
                ControllerAction.FORCE_ANSWER,
                None,
                "belief_stop_overrides_retrieval",
                None,
                snapshot.target_slot,
                snapshot,
            )

        should_inject = retrieval_calls >= self.config.injection_after_retrievals
        entity: Optional[str] = None
        if should_inject and self.config.uses_gold_entity:
            entity = _normalize_space(gold_entity or "") or None
            if entity is None:
                raise ValueError("gold-entity controller requires an oracle entity")
        elif should_inject and self.config.uses_gap and snapshot.bound_entities:
            entity = snapshot.bound_entities[0]

        query = str(policy_query or "")
        if policy_answer:
            query = snapshot.target_hypothesis or question
        if should_inject and (self.config.uses_gap or self.config.uses_gold_entity):
            query = build_gap_directed_query(
                query,
                snapshot.target_hypothesis if self.config.uses_gap else None,
                entity,
                strategy=(
                    self.config.query_strategy
                    if self.config.uses_gap
                    else "append_entity"
                ),
            )
        elif not query.strip():
            query = snapshot.target_hypothesis or question

        return ControllerDecisionV64(
            action=(
                ControllerAction.OVERRIDE_ANSWER_WITH_QUERY
                if policy_answer
                else ControllerAction.EXECUTE_QUERY
            ),
            query=query,
            reason=(
                "belief_rejects_low_coverage_answer"
                if policy_answer
                else "gap_directed_query"
                if query != policy_query
                else "policy_query_accepted"
            ),
            injected_entity=entity,
            target_slot=snapshot.target_slot,
            snapshot=snapshot,
        )


@dataclass(frozen=True)
class ControllerArtifactV64:
    config: ControllerConfigV64
    calibration_split_id_hash: Optional[str]
    expected_mean_retrievals: Optional[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONTROLLER_SCHEMA
    version: int = CONTROLLER_VERSION

    def __post_init__(self) -> None:
        if self.calibration_split_id_hash is not None and not str(
            self.calibration_split_id_hash
        ).strip():
            raise ValueError("controller calibration split hash cannot be empty")
        if self.expected_mean_retrievals is not None and (
            not math.isfinite(float(self.expected_mean_retrievals))
            or float(self.expected_mean_retrievals) < 0.0
        ):
            raise ValueError(
                "controller expected mean retrievals must be finite and non-negative"
            )
        calibration_ids = self.metadata.get("calibration_question_ids")
        if calibration_ids:
            observed_hash = hashlib.sha256(
                "\n".join(
                    sorted({str(value) for value in calibration_ids})
                ).encode("utf-8")
            ).hexdigest()
            if observed_hash != self.calibration_split_id_hash:
                raise ValueError(
                    "controller calibration question ids do not match split hash"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "config": {
                **asdict(self.config),
                "mode": self.config.mode.value,
            },
            "calibration_split_id_hash": self.calibration_split_id_hash,
            "expected_mean_retrievals": self.expected_mean_retrievals,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerArtifactV64":
        if payload.get("schema") != CONTROLLER_SCHEMA:
            raise ValueError("not an ACEC v6.4 controller artifact")
        if int(payload.get("version", -1)) != CONTROLLER_VERSION:
            raise ValueError("unsupported ACEC controller artifact version")
        config_payload = dict(payload["config"])
        config_payload["mode"] = ControllerMode(config_payload["mode"])
        return cls(
            config=ControllerConfigV64(**config_payload),
            calibration_split_id_hash=payload.get("calibration_split_id_hash"),
            expected_mean_retrievals=(
                None
                if payload.get("expected_mean_retrievals") is None
                else float(payload["expected_mean_retrievals"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


def save_controller_artifact_v64(
    path: str | Path, artifact: ControllerArtifactV64
) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite controller artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_controller_artifact_v64(path: str | Path) -> ControllerArtifactV64:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ControllerArtifactV64.from_dict(json.load(handle))


@dataclass(frozen=True)
class BridgeEntityLabelV64:
    question_id: str
    entity: Optional[str]
    assessable: bool
    basis: str
    supporting_titles: Tuple[str, ...]
    schema: str = BRIDGE_ENTITY_SCHEMA
    version: int = CONTROLLER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _context_pairs(record: Mapping[str, Any]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    context = record.get("context")
    if isinstance(context, Mapping):
        return tuple(
            (str(title), tuple(str(sentence) for sentence in sentences))
            for title, sentences in zip(
                context.get("title", []), context.get("sentences", [])
            )
        )
    if isinstance(context, list):
        return tuple(
            (str(item[0]), tuple(str(sentence) for sentence in item[1]))
            for item in context
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
    return ()


def _supporting_pairs(record: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
    supporting = record.get("supporting_facts") or {}
    if isinstance(supporting, Mapping):
        raw = zip(supporting.get("title", []), supporting.get("sent_id", []))
    else:
        raw = supporting
    return tuple((str(title), int(sent_id)) for title, sent_id in raw)


def _normal_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    return " ".join(re.findall(r"\w+", text))


def derive_gold_bridge_entity_v64(
    record: Mapping[str, Any]
) -> BridgeEntityLabelV64:
    """Derive an oracle bridge title with auditable, deterministic rules.

    The preferred rule follows a cross-title mention in the annotated
    supporting sentence.  A unique supporting title absent from the question
    is only a fallback.  Ambiguous cases are excluded rather than guessed.
    """

    qid = str(record.get("_id", record.get("id", "")))
    question_type = str(record.get("type", record.get("question_type", "")))
    supporting_pairs = _supporting_pairs(record)
    supporting_titles = tuple(dict.fromkeys(title for title, _ in supporting_pairs))
    if question_type and question_type != "bridge":
        return BridgeEntityLabelV64(
            qid, None, False, "not_bridge_type", supporting_titles
        )
    if len(supporting_titles) != 2:
        return BridgeEntityLabelV64(
            qid, None, False, "supporting_title_count_not_two", supporting_titles
        )

    context = {title: sentences for title, sentences in _context_pairs(record)}
    supporting_text: Dict[str, str] = {}
    for title, sent_id in supporting_pairs:
        sentences = context.get(title, ())
        if 0 <= sent_id < len(sentences):
            supporting_text[title] = (
                supporting_text.get(title, "") + " " + sentences[sent_id]
            )

    candidates = []
    for source in supporting_titles:
        source_text = _normal_text(supporting_text.get(source, ""))
        for target in supporting_titles:
            if source != target and _normal_text(target) in source_text:
                candidates.append(target)
    unique = tuple(dict.fromkeys(candidates))
    basis = "supporting_sentence_cross_title"
    if len(unique) != 1:
        normalized_question = _normal_text(str(record.get("question", "")))
        unmentioned = tuple(
            title
            for title in supporting_titles
            if _normal_text(title) not in normalized_question
        )
        unique = unmentioned
        basis = "unique_unmentioned_supporting_title"
    if len(unique) != 1:
        return BridgeEntityLabelV64(
            qid, None, False, "ambiguous_bridge_entity", supporting_titles
        )

    entity = unique[0]
    gold_answer = record.get("answer")
    if gold_answer is not None and _normal_text(entity) == _normal_text(str(gold_answer)):
        return BridgeEntityLabelV64(
            qid, None, False, "bridge_entity_equals_gold_answer", supporting_titles
        )
    return BridgeEntityLabelV64(qid, entity, True, basis, supporting_titles)


__all__ = [
    "BeliefInferenceControllerV64",
    "BeliefSnapshotV64",
    "BRIDGE_ENTITY_SCHEMA",
    "BridgeEntityLabelV64",
    "CONTROLLER_SCHEMA",
    "CONTROLLER_VERSION",
    "ControllerAction",
    "ControllerArtifactV64",
    "ControllerConfigV64",
    "ControllerDecisionV64",
    "ControllerMode",
    "FORCE_ANSWER_INSTRUCTION",
    "SLOT_CONTRACT_VERSION",
    "SlotDecompositionV64",
    "build_gap_directed_query",
    "derive_gold_bridge_entity_v64",
    "load_controller_artifact_v64",
    "parse_slot_decomposition",
    "preseed_belief_slots",
    "save_controller_artifact_v64",
    "slot_decomposition_prompt",
]
