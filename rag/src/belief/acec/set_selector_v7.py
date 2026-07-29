"""State- and set-conditioned learnable selector for ACEC v7.

The selector never deletes a candidate for a semantic reason.  Every valid,
unique candidate remains in the Plackett-Luce support until it is selected.
Contradiction, low relevance, and apparent redundancy are numerical features
whose sign and importance are learned from downstream answer utility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .answer_value_v7 import AnswerValueArtifactV7
from .belief_simulator_v7 import BeliefSimulatorV7
from .contracts_v7 import (
    CandidateV7,
    SelectionStepV7,
    SelectionTraceV7,
    SelectorStateV7,
    assert_sf_label_free,
    state_fingerprint_v7,
)


SELECTOR_SCHEMA_V7 = "acec_learnable_set_selector"
SELECTOR_VERSION_V7 = 70

SELECTOR_FEATURE_NAMES_V7: Tuple[str, ...] = (
    "state_coverage",
    "state_coverage_std",
    "state_k_entropy",
    "state_turn_fraction",
    "state_budget_fraction",
    "state_target_probability",
    "state_mean_slot_probability",
    "state_bound_fraction",
    "retrieval_score_z",
    "reciprocal_rank",
    "conditional_coverage_gain",
    "information_gain",
    "signed_answer_value",
    "max_entailment",
    "max_contradiction",
    "max_neutral",
    "target_slot_hit",
    "selected_slot_overlap",
    "log_document_length",
    "source_quality",
)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("selector temperature must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("selector produced a non-finite score")
    scaled = values.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probabilities = np.maximum(np.exp(np.clip(scaled, -700.0, 0.0)), 1e-12)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return probabilities / total


def _robust_z(values: Sequence[float]) -> Dict[int, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    q75, q25 = np.percentile(array, [75, 25])
    scale = float(q75 - q25)
    if scale <= 1e-9:
        standard = float(np.std(array))
        scale = standard if standard > 1e-9 else 1.0
    return {index: float((value - median) / scale) for index, value in enumerate(array)}


def _selected_overlap(
    candidate: CandidateV7,
    selected_ids: Sequence[str],
    candidates: Mapping[str, CandidateV7],
) -> float:
    if not selected_ids:
        return 0.0
    vector = np.asarray(candidate.slot_entailment, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    best = 0.0
    for candidate_id in selected_ids:
        other = np.asarray(candidates[candidate_id].slot_entailment, dtype=np.float64)
        denominator = norm * float(np.linalg.norm(other))
        similarity = float(vector @ other / denominator) if denominator > 1e-12 else 0.0
        best = max(best, similarity)
    return best


def base_answer_value_features_v7(
    state: SelectorStateV7,
    candidate: CandidateV7,
    simulator: BeliefSimulatorV7,
    selected_ids: Sequence[str],
    *,
    retrieval_score_z: float,
) -> Dict[str, float]:
    """Compact independent feature set consumed by a signed value artifact."""

    conditional = simulator.conditional_metrics(candidate.candidate_id, selected_ids)
    return {
        "state_coverage": float(state.coverage),
        "state_uncertainty": float(state.coverage_std),
        "retrieval_score_z": float(retrieval_score_z),
        "conditional_coverage_gain": float(
            conditional["conditional_coverage_gain"]
        ),
        "information_gain": float(conditional["information_gain"]),
        "max_entailment": max(candidate.slot_entailment, default=0.0),
        "max_contradiction": max(candidate.slot_contradiction, default=0.0),
        "target_slot_hit": (
            float(candidate.slot_hit_probabilities[state.target_slot])
            if state.target_slot is not None
            else max(candidate.slot_hit_probabilities, default=0.0)
        ),
        "source_quality": float(candidate.source_quality),
    }


def selector_features_v7(
    state: SelectorStateV7,
    candidate: CandidateV7,
    simulator: BeliefSimulatorV7,
    selected_ids: Sequence[str],
    *,
    retrieval_score_z: float,
    answer_value_artifact: Optional[AnswerValueArtifactV7] = None,
) -> Dict[str, float]:
    conditional = simulator.conditional_metrics(candidate.candidate_id, selected_ids)
    target_probability = (
        state.slot_probabilities[state.target_slot]
        if state.target_slot is not None
        else min(state.slot_probabilities)
    )
    target_hit = (
        candidate.slot_hit_probabilities[state.target_slot]
        if state.target_slot is not None
        else max(candidate.slot_hit_probabilities)
    )
    signed_value = float(candidate.signed_answer_value)
    if answer_value_artifact is not None:
        signed_value = answer_value_artifact.score(
            base_answer_value_features_v7(
                state,
                candidate,
                simulator,
                selected_ids,
                retrieval_score_z=retrieval_score_z,
            )
        )
    contradiction = (
        max(candidate.slot_contradiction)
        if candidate.slot_contradiction
        else 0.0
    )
    neutral = max(candidate.slot_neutral) if candidate.slot_neutral else 0.0
    values = {
        "state_coverage": float(state.coverage),
        "state_coverage_std": float(state.coverage_std),
        "state_k_entropy": float(state.k_entropy),
        "state_turn_fraction": min(
            float(state.turn_index) / max(state.max_turns, 1), 1.0
        ),
        "state_budget_fraction": min(
            float(state.retrieval_budget_remaining) / max(state.max_turns, 1),
            1.0,
        ),
        "state_target_probability": float(target_probability),
        "state_mean_slot_probability": float(
            np.mean(state.slot_probabilities)
        ),
        "state_bound_fraction": float(np.mean(state.slot_bound)),
        "retrieval_score_z": float(retrieval_score_z),
        "reciprocal_rank": 1.0 / float(candidate.retrieval_rank + 1),
        "conditional_coverage_gain": float(
            conditional["conditional_coverage_gain"]
        ),
        "information_gain": float(conditional["information_gain"]),
        "signed_answer_value": signed_value,
        "max_entailment": max(candidate.slot_entailment, default=0.0),
        "max_contradiction": contradiction,
        "max_neutral": neutral,
        "target_slot_hit": float(target_hit),
        "selected_slot_overlap": _selected_overlap(
            candidate, selected_ids, simulator.candidates
        ),
        "log_document_length": math.log1p(candidate.document_length),
        "source_quality": float(candidate.source_quality),
    }
    if tuple(values) != SELECTOR_FEATURE_NAMES_V7:
        raise AssertionError("selector feature contract order changed")
    return values


@dataclass(frozen=True)
class SelectorArtifactV7:
    feature_names: Tuple[str, ...]
    center: Tuple[float, ...]
    scale: Tuple[float, ...]
    hidden_weights: Tuple[Tuple[float, ...], ...]
    hidden_bias: Tuple[float, ...]
    output_weights: Tuple[float, ...]
    output_bias: float
    temperature: float
    fit_question_id_sha256: str
    validation_question_id_sha256: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SELECTOR_SCHEMA_V7
    version: int = SELECTOR_VERSION_V7

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        hidden = len(self.hidden_bias)
        if tuple(self.feature_names) != SELECTOR_FEATURE_NAMES_V7:
            raise ValueError("selector artifact feature contract differs from v7")
        if not (len(self.center) == len(self.scale) == size):
            raise ValueError("selector normalization vectors do not align")
        if len(self.hidden_weights) != hidden or len(self.output_weights) != hidden:
            raise ValueError("selector hidden/output dimensions do not align")
        if any(len(row) != size for row in self.hidden_weights):
            raise ValueError("selector hidden weight row has wrong width")
        if any(float(value) <= 0.0 for value in self.scale):
            raise ValueError("selector feature scale must be positive")
        numeric_values = [
            *self.center,
            *self.scale,
            *self.hidden_bias,
            *self.output_weights,
            self.output_bias,
            *(value for row in self.hidden_weights for value in row),
        ]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("selector artifact contains non-finite parameters")
        if not all(
            math.isfinite(float(value)) for value in self.metrics.values()
        ):
            raise ValueError("selector artifact contains non-finite metrics")
        if self.temperature <= 0.0:
            raise ValueError("selector temperature must be positive")
        if self.fit_question_id_sha256 == self.validation_question_id_sha256:
            raise ValueError("selector fit and validation hashes must differ")
        disabled = {
            str(value)
            for value in self.metadata.get("disabled_features", ())
        }
        unknown_disabled = disabled - set(self.feature_names)
        if unknown_disabled:
            raise ValueError(
                f"selector artifact disables unknown features: {sorted(unknown_disabled)}"
            )
        assert_sf_label_free(self.metadata, path="selector.metadata")

    def score(self, features: Mapping[str, float]) -> float:
        if set(features) != set(self.feature_names):
            raise ValueError("selector features do not match artifact order")
        disabled = {
            str(value)
            for value in self.metadata.get("disabled_features", ())
        }
        vector = np.asarray(
            [
                0.0 if name in disabled else float(features[name])
                for name in self.feature_names
            ],
            dtype=np.float64,
        )
        normalized = (vector - np.asarray(self.center)) / np.asarray(self.scale)
        hidden = np.maximum(
            np.asarray(self.hidden_weights) @ normalized
            + np.asarray(self.hidden_bias),
            0.0,
        )
        return float(np.asarray(self.output_weights) @ hidden + self.output_bias)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectorArtifactV7":
        if payload.get("schema") != SELECTOR_SCHEMA_V7:
            raise ValueError("not an ACEC v7 selector artifact")
        return cls(
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            center=tuple(float(v) for v in payload["center"]),
            scale=tuple(float(v) for v in payload["scale"]),
            hidden_weights=tuple(
                tuple(float(v) for v in row)
                for row in payload["hidden_weights"]
            ),
            hidden_bias=tuple(float(v) for v in payload["hidden_bias"]),
            output_weights=tuple(float(v) for v in payload["output_weights"]),
            output_bias=float(payload["output_bias"]),
            temperature=float(payload["temperature"]),
            fit_question_id_sha256=str(payload["fit_question_id_sha256"]),
            validation_question_id_sha256=str(
                payload["validation_question_id_sha256"]
            ),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
            schema=str(payload["schema"]),
            version=int(payload.get("version", SELECTOR_VERSION_V7)),
        )

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def initialize_selector_artifact_v7(
    *,
    fit_question_id_sha256: str,
    validation_question_id_sha256: str,
    hidden_size: int = 16,
    seed: int = 0,
    temperature: float = 0.7,
) -> SelectorArtifactV7:
    """Create a deterministic small artifact for smoke tests/pretraining."""

    if hidden_size <= 0:
        raise ValueError("hidden size must be positive")
    rng = np.random.default_rng(seed)
    width = len(SELECTOR_FEATURE_NAMES_V7)
    hidden = rng.normal(0.0, 0.02, size=(hidden_size, width))
    output = rng.normal(0.0, 0.02, size=hidden_size)
    return SelectorArtifactV7(
        feature_names=SELECTOR_FEATURE_NAMES_V7,
        center=(0.0,) * width,
        scale=(1.0,) * width,
        hidden_weights=tuple(tuple(float(v) for v in row) for row in hidden),
        hidden_bias=(0.0,) * hidden_size,
        output_weights=tuple(float(v) for v in output),
        output_bias=0.0,
        temperature=temperature,
        fit_question_id_sha256=fit_question_id_sha256,
        validation_question_id_sha256=validation_question_id_sha256,
        metrics={"initialized_only": 1.0},
        metadata={"initialized_only": True, "evidence_membership_supervision_used": False},
    )


class SequentialSetSelectorV7:
    def __init__(
        self,
        *,
        artifact: Optional[SelectorArtifactV7],
        answer_value_artifact: Optional[AnswerValueArtifactV7] = None,
        strategy: str = "learned",
        fixed_coverage_weight: float = 1.0,
    ) -> None:
        if strategy not in {"learned", "relevance", "fixed_rel_plus_coverage"}:
            raise ValueError(f"unsupported v7 selector strategy {strategy!r}")
        if strategy == "learned" and artifact is None:
            raise ValueError("learned selector requires an artifact")
        self.artifact = artifact
        self.answer_value_artifact = answer_value_artifact
        self.strategy = strategy
        self.fixed_coverage_weight = float(fixed_coverage_weight)

    def _score(
        self,
        features: Mapping[str, float],
    ) -> float:
        if self.strategy == "relevance":
            return float(features["retrieval_score_z"])
        if self.strategy == "fixed_rel_plus_coverage":
            return float(features["retrieval_score_z"]) + self.fixed_coverage_weight * float(
                features["conditional_coverage_gain"]
            )
        assert self.artifact is not None
        return self.artifact.score(features)

    def select(
        self,
        state: SelectorStateV7,
        candidates: Sequence[CandidateV7],
        *,
        k: int,
        sample: bool = False,
        seed: int = 0,
        temperature: Optional[float] = None,
    ) -> Tuple[Tuple[CandidateV7, ...], SelectionTraceV7]:
        if k <= 0:
            raise ValueError("selector K must be positive")
        if not candidates:
            return (), SelectionTraceV7(
                question_id=state.question_id,
                turn_index=state.turn_index,
                strategy=self.strategy,
                pool_ids=(),
                selected_ids=(),
                steps=(),
                candidate_pool_size=0,
                selected_k=k,
                selector_artifact_sha256=(
                    self.artifact.sha256 if self.artifact is not None else "baseline"
                ),
                answer_value_artifact_sha256=(
                    self.answer_value_artifact.sha256
                    if self.answer_value_artifact is not None
                    else "none"
                ),
                state_fingerprint_before=state_fingerprint_v7(state),
            )
        simulator = BeliefSimulatorV7(state, candidates)
        score_z = _robust_z([candidate.retrieval_score for candidate in candidates])
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        remaining = list(candidates)
        selected_ids: List[str] = []
        selected: List[CandidateV7] = []
        steps: List[SelectionStepV7] = []
        rng = np.random.default_rng(seed)
        effective_temperature = (
            float(temperature)
            if temperature is not None
            else (self.artifact.temperature if self.artifact is not None else 0.7)
        )
        while remaining and len(selected) < k:
            feature_rows = [
                selector_features_v7(
                    state,
                    candidate,
                    simulator,
                    selected_ids,
                    retrieval_score_z=score_z[
                        next(
                            index
                            for index, original in enumerate(candidates)
                            if original.candidate_id == candidate.candidate_id
                        )
                    ],
                    answer_value_artifact=self.answer_value_artifact,
                )
                for candidate in remaining
            ]
            scores = np.asarray(
                [self._score(features) for features in feature_rows],
                dtype=np.float64,
            )
            probabilities = _softmax(scores, effective_temperature)
            available_before_choice = list(remaining)
            chosen_index = (
                int(rng.choice(len(remaining), p=probabilities))
                if sample
                else int(np.argmax(scores))
            )
            chosen = remaining.pop(chosen_index)
            chosen_features = feature_rows[chosen_index]
            selected_ids.append(chosen.candidate_id)
            selected.append(chosen)
            probability_map = {
                candidate.candidate_id: float(probability)
                for candidate, probability in zip(
                    available_before_choice, probabilities
                )
            }
            chosen_probability = probability_map[chosen.candidate_id]
            steps.append(
                SelectionStepV7(
                    position=len(selected) - 1,
                    selected_id=chosen.candidate_id,
                    log_probability=math.log(max(chosen_probability, 1e-12)),
                    score=float(scores[chosen_index]),
                    conditional_coverage_gain=float(
                        chosen_features["conditional_coverage_gain"]
                    ),
                    information_gain=float(chosen_features["information_gain"]),
                    signed_answer_value=float(
                        chosen_features["signed_answer_value"]
                    ),
                    candidate_probabilities=probability_map,
                )
            )
        trace = SelectionTraceV7(
            question_id=state.question_id,
            turn_index=state.turn_index,
            strategy=self.strategy,
            pool_ids=tuple(candidate.candidate_id for candidate in candidates),
            selected_ids=tuple(selected_ids),
            steps=tuple(steps),
            candidate_pool_size=len(candidates),
            selected_k=k,
            selector_artifact_sha256=(
                self.artifact.sha256 if self.artifact is not None else "baseline"
            ),
            answer_value_artifact_sha256=(
                self.answer_value_artifact.sha256
                if self.answer_value_artifact is not None
                else "none"
            ),
            state_fingerprint_before=state_fingerprint_v7(state),
        )
        # The dictionary is also a structural assertion that candidate ids are
        # unique; BeliefSimulatorV7 already fails closed on duplicates.
        if set(by_id) != set(trace.pool_ids):
            raise AssertionError("selector pool changed during selection")
        return tuple(selected), trace


def save_selector_artifact_v7(path: Path, artifact: SelectorArtifactV7) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite selector artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_selector_artifact_v7(path: Path) -> SelectorArtifactV7:
    return SelectorArtifactV7.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


__all__ = [
    "SELECTOR_FEATURE_NAMES_V7",
    "SELECTOR_SCHEMA_V7",
    "SelectorArtifactV7",
    "SequentialSetSelectorV7",
    "base_answer_value_features_v7",
    "initialize_selector_artifact_v7",
    "load_selector_artifact_v7",
    "save_selector_artifact_v7",
    "selector_features_v7",
]
